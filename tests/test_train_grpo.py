"""Tests for scripts/train_grpo.py's model-free parts (CLAUDE.md §9, §12).

The training loop needs the weights and the GPU (graphics processing unit) and is exercised by running the
script with --limit/--max-steps. These pin the reward -- the four §9 states,
the brevity bonus paid only on a match, truncation as a format failure, the
timeout retry, one execution per distinct command -- and that the policy is
trained on the prompt it is served and judged by the harness it is measured by.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import train_grpo  # noqa: E402
from prompt import messages_for_case  # noqa: E402

EOS = 7
CASE = {"id": "x-1", "category": "substitute", "tier": "T1", "mode": "ex",
        "instruction": "replace cat with dog everywhere",
        "buffer_before": "cat\n", "buffer_after": "dog\n",
        "gold_command": "%s/cat/dog/g", "cursor": [1, 1]}


def _assistant(text):
    return [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------- lengths

def test_completion_length_excludes_the_end_of_turn_token():
    """§11 charges content tokens; the stop token is not content. That is also
    what baseline.generate_predictions counts, so the brevity bonus and the
    results table measure the same thing."""
    assert train_grpo.completion_length([1, 2, 3, EOS], EOS) == (3, True)
    assert train_grpo.completion_length([1, 2, 3], EOS) == (3, False)
    assert train_grpo.completion_length([], EOS) == (0, False)


# ----------------------------------------------------------------- reward

def test_reward_table_is_section_9():
    cap = 192
    assert train_grpo.reward_for("parse", 3, cap) == 0.0
    assert train_grpo.reward_for("truncated", 192, cap) == 0.0
    assert train_grpo.reward_for("invalid", 3, cap) == 0.0
    assert train_grpo.reward_for("valid_wrong", 3, cap) == 0.1
    assert train_grpo.reward_for("match", 0, cap) == pytest.approx(1.2)
    assert train_grpo.reward_for("match", 96, cap) == pytest.approx(1.1)
    assert train_grpo.reward_for("match", 192, cap) == pytest.approx(1.0)


def test_brevity_bonus_is_paid_only_on_a_match():
    """The anti-reward-hacking term: a one-token wrong answer is worth exactly
    0.1 and a one-token unparseable one exactly 0, so brevity can never be
    collected without correctness."""
    assert train_grpo.reward_for("valid_wrong", 1, 192) == 0.1
    assert train_grpo.reward_for("parse", 1, 192) == 0.0
    assert train_grpo.reward_for("match", 1, 192) > train_grpo.reward_for("match", 50, 192)


def test_brevity_bonus_never_goes_negative():
    assert train_grpo.reward_for("match", 500, 192) == pytest.approx(1.0)


# --------------------------------------------------------------- classify

def test_truncated_completion_is_a_format_failure_before_parsing():
    """A completion that ran into the cap never stopped, so §6's "exactly one
    line" is not met even if its first line parses."""
    state, mode, cmd, n = train_grpo.classify("ex: %s/a/b/\nmore", [1, 2, 3, 4], EOS)
    assert state == "truncated" and mode is None and cmd is None and n == 4


def test_parse_failure_and_think_leak_score_as_parse():
    assert train_grpo.classify("I would use :s", [1, EOS], EOS)[0] == "parse"
    assert train_grpo.classify("<think>hm</think>\nex: %d", [1, EOS], EOS)[0] == "parse"


def test_parsed_completion_defers_to_execution():
    state, mode, cmd, n = train_grpo.classify("keys: dw", [1, 2, EOS], EOS)
    assert state is None and (mode, cmd) == ("keys", "dw") and n == 2


# --------------------------------------------------------- execution state

def _scorer(entries):
    """A fake evaluate.score: pops one per_case entry per call and counts."""
    calls = []

    def score(cases, preds, timeout):
        calls.append((cases[0]["id"], preds[cases[0]["id"]]["command"], timeout))
        e = entries.pop(0)
        return {"per_case": {cases[0]["id"]: e}}
    score.calls = calls
    return score


def test_execution_state_names_the_harness_verdict():
    for entry, expected in (({"match": True, "valid": True, "timeout": False}, "match"),
                            ({"match": False, "valid": True, "timeout": False}, "valid_wrong"),
                            ({"match": False, "valid": False, "timeout": False}, "invalid")):
        sc = _scorer([entry])
        assert train_grpo.execution_state(CASE, "ex", "%d", 5.0, scorer=sc) == expected
        assert sc.calls == [("x-1", "%d", 5.0)]


def test_timeout_is_retried_once_before_it_scores_zero():
    """HANDOFF.md: the 5 s limit has fired spuriously under parallel load, and
    during GRPO that would score a correct command as 0.0."""
    sc = _scorer([{"match": False, "valid": False, "timeout": True},
                  {"match": True, "valid": True, "timeout": False}])
    assert train_grpo.execution_state(CASE, "ex", "%d", 5.0, scorer=sc) == "match"
    assert len(sc.calls) == 2
    sc = _scorer([{"match": False, "valid": False, "timeout": True},
                  {"match": False, "valid": False, "timeout": True},
                  {"match": True, "valid": True, "timeout": False}])   # never reached
    assert train_grpo.execution_state(CASE, "ex", "%d", 5.0, scorer=sc) == "invalid"
    assert len(sc.calls) == 2, "one retry, not unbounded"


def test_execution_state_uses_the_real_harness():
    """No fake: the reward's notion of a match is evaluate.score's."""
    assert train_grpo.execution_state(CASE, "ex", "%s/cat/dog/g", 5.0) == "match"
    assert train_grpo.execution_state(CASE, "ex", "%s/cat/cow/g", 5.0) == "valid_wrong"
    assert train_grpo.execution_state(CASE, "ex", "%s/xyz/dog/g", 5.0) == "invalid"


# ------------------------------------------------------------ reward func

def test_reward_function_scores_a_batch_and_executes_each_command_once(monkeypatch):
    calls = []

    def fake_state(case, mode, command, timeout):
        calls.append((case["id"], mode, command))
        return {"%s/cat/dog/g": "match", "%s/cat/cow/g": "valid_wrong",
                "%s/x/y/": "invalid"}[command]
    monkeypatch.setattr(train_grpo, "execution_state", fake_state)
    fn = train_grpo.make_reward(EOS, len_cap=192, timeout=5.0, workers=2)
    completions = [_assistant("ex: %s/cat/dog/g"), _assistant("ex: %s/cat/dog/g"),
                   _assistant("ex: %s/cat/cow/g"), _assistant("ex: %s/x/y/"),
                   _assistant("nope"), _assistant("ex: %s/cat/dog/g")]
    ids = [[1, 2, 3, EOS], [1, 2, 3, EOS], [1, 2, 3, EOS], [1, 2, EOS], [1, EOS], [1, 2, 3]]
    logged, extra = {}, {}
    rewards = fn(prompts=[None] * 6, completions=completions, completion_ids=ids,
                 id=["x-1"] * 6, buffer_before=["cat\n"] * 6, buffer_after=["dog\n"] * 6,
                 cursor=[[1, 1]] * 6,
                 log_metric=lambda k, v: logged.__setitem__(k, v),
                 log_extra=lambda k, v: extra.__setitem__(k, v))
    assert rewards == pytest.approx([1.0 + 0.2 * (1 - 3 / 192)] * 2 + [0.1, 0.0, 0.0, 0.0])
    assert extra["state"] == ["match", "match", "valid_wrong", "invalid", "parse", "truncated"]
    # three distinct commands were executed, once each -- not five
    assert sorted(c[2] for c in calls) == ["%s/cat/cow/g", "%s/cat/dog/g", "%s/x/y/"]
    assert logged["reward_state/match"] == pytest.approx(2 / 6)
    assert logged["reward_state/truncated"] == pytest.approx(1 / 6)
    assert sum(logged["reward_state/" + s] for s in train_grpo.REWARD_STATES) == pytest.approx(1.0)
    assert logged["reward_state/distinct_commands"] == pytest.approx(3 / 6)


def test_reward_function_judges_by_the_case_buffers_not_by_a_gold_command():
    """Two prompts, same command: right for one buffer, wrong for the other."""
    fn = train_grpo.make_reward(EOS, len_cap=192, timeout=5.0, workers=2)
    rewards = fn(prompts=[None, None],
                 completions=[_assistant("ex: %s/cat/dog/g")] * 2,
                 completion_ids=[[1, 2, EOS]] * 2,
                 id=["a", "b"], buffer_before=["cat\n", "cat\n"],
                 buffer_after=["dog\n", "cow\n"], cursor=[[1, 1], [1, 1]])
    assert rewards[0] > 1.0 and rewards[1] == 0.1


# ------------------------------------------------------------------- data

def test_dataset_is_the_inference_prompt_plus_what_the_reward_needs():
    pytest.importorskip("datasets")
    ds = train_grpo.build_dataset([CASE])
    row = ds[0]
    assert row["prompt"] == messages_for_case(CASE)
    assert row["prompt"][-1]["role"] == "user"
    assert {"id", "buffer_before", "buffer_after", "cursor", "category"} <= set(ds.column_names)
    assert row["cursor"] == [1, 1]
    # the reward judges by execution; the answer must not ride along
    assert "gold_command" not in ds.column_names and "completion" not in ds.column_names
    assert "chat_template_kwargs" not in ds.column_names     # config field, not a row


def test_dataset_defaults_the_cursor():
    pytest.importorskip("datasets")
    row = train_grpo.build_dataset([{k: v for k, v in CASE.items() if k != "cursor"}])[0]
    assert row["cursor"] == [1, 1]


# ------------------------------------------------------------- contract

def test_defaults_are_the_contract_in_section_9():
    d = {a.dest: a.default for a in train_grpo.build_arg_parser()._actions}
    assert 6 <= d["num_generations"] <= 8
    assert d["beta"] == 0.0                              # library default
    assert d["reward_workers"] == 8 and d["timeout"] == 5.0
    assert d["max_hours"] == 12.0                        # §14b hard stop
    assert d["max_completion_length"] == 192             # baseline.py's cap
    assert d["scale_rewards"] == "batch" and d["loss_type"] == "dapo"
    assert d["temperature"] == 1.0 and d["lr"] == 1e-5
    assert (train_grpo.REWARD_MATCH, train_grpo.REWARD_VALID_WRONG,
            train_grpo.BREVITY_BONUS) == (1.0, 0.1, 0.2)


def test_one_generation_and_one_evaluation_path_shared_with_the_results_table():
    """The val callback, the final full-val pass and the base loader all come
    from the SFT/baseline scripts; a private copy here would let the GRPO
    row drift from every other row."""
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert "from train_sft import" in src and "make_callbacks" in src
    assert "full_val_eval(" in src and 'condition="grpo"' in src
    assert "def generate_predictions" not in src and "def score(" not in src
    assert "AutoModelForCausalLM" not in src and src.count("load_base(") == 1
    assert "write_base_marker(" in src                   # final/ says which base
    body = src.split("def main()", 1)[1]
    assert ".generate(" not in body, "sampling is TRL's; greedy evals are baseline's"


def test_no_model_ids_and_no_thinking_here():
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert "Qwen/" not in src and "hf_id: " not in src
    assert 'reg["hf_id"]' in src
    assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src


def test_cli_help_needs_no_torch():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "train_grpo.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    for flag in ("--model", "--adapter", "--inspect", "--num-generations",
                 "--scale-rewards", "--beta", "--max-hours"):
        assert flag in p.stdout


def test_baseline_can_label_the_row():
    """The GRPO test row goes through baseline.py like every other row; the
    summary must say what it is and the run directory must not collide with
    the SFT row's."""
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "baseline.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--condition" in p.stdout
    src = (REPO / "scripts" / "baseline.py").read_text(encoding="utf-8")
    assert 'args.condition or ("sft" if args.adapter else "zero-shot")' in src
    assert '"-" + args.condition if args.adapter' in src


# --------------------------------------------------------- checkpoint resume

def _complete_checkpoint(path):
    path.mkdir()
    for name in train_grpo.CHECKPOINT_COMPLETE:
        (path / name).write_text("x")


def test_latest_checkpoint_picks_the_highest_complete_step(tmp_path):
    assert train_grpo.latest_checkpoint(tmp_path) is None
    for n in (50, 100, 9):
        _complete_checkpoint(tmp_path / ("checkpoint-%d" % n))
    (tmp_path / "checkpoint-junk").mkdir()
    (tmp_path / "checkpoint-200").write_text("a file, not a checkpoint")
    ck = train_grpo.latest_checkpoint(tmp_path)
    assert ck.name == "checkpoint-100" and train_grpo.checkpoint_step(ck) == 100


def test_resume_rebuilds_the_harness_selection_from_its_history(tmp_path):
    """An interrupted run's eval rows past the checkpoint belong to steps that
    will be trained again; the best-so-far must come from the rows that stand,
    with the same first-wins tie rule the live callback applies."""
    import json
    import train_sft
    from transformers import TrainerCallback
    HarnessEval, _ = train_sft.make_callbacks(TrainerCallback)
    ev = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)
    rows = [{"step": 0, "exec_match_pct": 45.2}, {"step": 50, "exec_match_pct": 46.0},
            {"step": 100, "exec_match_pct": 46.0}, {"step": 150, "exec_match_pct": 47.0}]
    with (tmp_path / "eval_history.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    kept = ev.restore(100)
    assert [r["step"] for r in kept] == [0, 50, 100]
    assert (ev.best, ev.best_step) == (46.0, 50)          # first of the tie, as in run()
    on_disk = [json.loads(l) for l in (tmp_path / "eval_history.jsonl").read_text().splitlines()]
    assert [r["step"] for r in on_disk] == [0, 50, 100], "rows past the checkpoint are dropped"
    assert ev.history == kept


def test_resume_with_no_history_starts_selection_from_scratch(tmp_path):
    import train_sft
    from transformers import TrainerCallback
    HarnessEval, _ = train_sft.make_callbacks(TrainerCallback)
    ev = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)
    assert ev.restore(100) == []
    assert (ev.best, ev.best_step) == (-1.0, None)


def test_checkpoints_are_written_and_resume_is_a_flag():
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert 'save_strategy="steps"' in src and "save_total_limit=2" in src
    assert "trainer.train(resume_from_checkpoint=" in src
    assert "evaluator.restore(" in src
    d = {a.dest: a.default for a in train_grpo.build_arg_parser()._actions}
    assert d["resume"] is False and d["save_every"] is None    # default: --eval-every


# ------------------------------------------------- review fixes (2026-09-13)

def test_incomplete_checkpoint_is_skipped_never_resumed_from(tmp_path, capsys):
    """transformers writes trainer_state.json last; a kill mid-write leaves a
    directory the Trainer cannot resume from, and the driver would stop
    retrying on the resulting error."""
    for step in (100, 150):
        (tmp_path / ("checkpoint-%d" % step)).mkdir()
    for name in train_grpo.CHECKPOINT_COMPLETE:
        (tmp_path / "checkpoint-100" / name).write_text("x")
    (tmp_path / "checkpoint-150" / "adapter_model.safetensors").write_text("x")   # no state
    ck = train_grpo.latest_checkpoint(tmp_path)
    assert ck.name == "checkpoint-100"
    assert "checkpoint-150" in capsys.readouterr().out
    assert (tmp_path / "checkpoint-150").exists(), "never deleted"
    assert train_grpo.checkpoint_complete(tmp_path / "checkpoint-100")
    assert not train_grpo.checkpoint_complete(tmp_path / "checkpoint-150")


def test_resume_orphans_a_best_adapter_from_a_step_past_the_checkpoint(tmp_path):
    """The eval at step N runs before checkpoint-N is written; a kill in
    between leaves best/ from a step the resumed run will train again."""
    import json
    import train_sft
    from transformers import TrainerCallback
    HarnessEval, _ = train_sft.make_callbacks(TrainerCallback)
    ev = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)
    with (tmp_path / "eval_history.jsonl").open("w") as fh:
        for r in ({"step": 0, "exec_match_pct": 45.2}, {"step": 50, "exec_match_pct": 46.0},
                  {"step": 100, "exec_match_pct": 47.0}):
            fh.write(json.dumps(r) + "\n")
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "selection.json").write_text(json.dumps({"step": 100, "value": 47.0}))
    (tmp_path / "best" / "adapter_model.safetensors").write_text("weights")
    ev.restore(50)
    assert not (tmp_path / "best").exists()
    assert (tmp_path / "best.orphaned-100" / "adapter_model.safetensors").exists()
    assert (ev.best, ev.best_step) == (46.0, 50)
    # a best/ from a kept step is left alone
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "selection.json").write_text(json.dumps({"step": 50, "value": 46.0}))
    ev.restore(50)
    assert (tmp_path / "best" / "selection.json").exists()


def test_last_window_is_evaluated_when_the_run_length_is_off_the_cadence(tmp_path):
    """max_steps 818 with evals every 50 would leave steps 801-818 trained but
    never scored, and final/ could never be selected."""
    import train_sft
    from transformers import TrainerCallback
    from types import SimpleNamespace
    HarnessEval, _ = train_sft.make_callbacks(TrainerCallback)
    ev = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)
    seen = []
    ev.run = lambda model, step: seen.append(step)
    ev.on_train_end(None, SimpleNamespace(global_step=818), None, model="m")
    ev.on_train_end(None, SimpleNamespace(global_step=800), None, model="m")   # on cadence
    ev.on_train_end(None, SimpleNamespace(global_step=0), None, model="m")     # nothing trained
    assert seen == [818]


def test_group_metrics_report_signal_per_prompt_group():
    """TRL's frac_reward_zero_std is a BATCH statistic under
    scale_rewards="batch"; these are the per-group numbers."""
    ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    states = (["match"] * 4                                   # all match, same length
              + ["match", "valid_wrong", "invalid", "parse"]  # mixed
              + ["valid_wrong"] * 4)                          # no match, identical
    rewards = [1.19] * 4 + [1.19, 0.1, 0.0, 0.0] + [0.1] * 4
    m = train_grpo.group_metrics(ids, states, rewards)
    assert m == pytest.approx({"zero_signal_groups": 2 / 3, "all_match_groups": 1 / 3,
                               "no_match_groups": 1 / 3})
    # an all-match group whose brevity bonuses differ is not zero-signal
    m = train_grpo.group_metrics(["a"] * 2, ["match"] * 2, [1.19, 1.18])
    assert m["zero_signal_groups"] == 0.0 and m["all_match_groups"] == 1.0


def test_reward_function_logs_group_metrics_and_keys_share(monkeypatch):
    monkeypatch.setattr(train_grpo, "execution_state",
                        lambda case, mode, command, timeout: "match" if mode == "keys" else "invalid")
    fn = train_grpo.make_reward(EOS, len_cap=192, timeout=5.0, workers=2)
    completions = [_assistant("keys: dw"), _assistant("keys: dw"),
                   _assistant("ex: %d"), _assistant("nope")]
    ids = [[1, 2, EOS], [1, 2, EOS], [1, 2, EOS], [1, EOS]]
    logged = {}
    fn(prompts=[None] * 4, completions=completions, completion_ids=ids,
       id=["a", "a", "b", "b"], buffer_before=["x\n"] * 4, buffer_after=["y\n"] * 4,
       cursor=[[1, 1]] * 4, log_metric=lambda k, v: logged.__setitem__(k, v))
    assert logged["reward_state/keys_share"] == pytest.approx(0.5)      # parse failures count as neither
    assert logged["reward_state/zero_signal_groups"] == pytest.approx(1.0)   # a: both 1.19; b: both 0
    assert logged["reward_state/all_match_groups"] == pytest.approx(0.5)
    assert logged["reward_state/no_match_groups"] == pytest.approx(0.5)


def test_completion_text_accepts_a_plain_string():
    assert train_grpo.completion_text("ex: %d") == "ex: %d"
    assert train_grpo.completion_text(_assistant("ex: %d")) == "ex: %d"


def test_console_table_and_parquet_trail_are_independent():
    """--print-completions 0 silences the console table only; log_completions
    stays on so the parquet audit trail is always written."""
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert "log_completions=True" in src
    assert "cfg.log_completions = False" not in src
    assert "num_completions_to_print=args.print_completions" in src


def test_run_name_and_resume_config_are_separate_from_the_out_dir():
    d = {a.dest: a.default for a in train_grpo.build_arg_parser()._actions}
    assert d["run_name"] is None                       # -> <model>-grpo
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert 'args.run_name or reg["name"] + "-grpo"' in src
    assert 'train_config.resume-%d.json' in src         # a resume never overwrites the recipe


# ------------------------------------------------------ graceful pause (2026-09-13)

def _pauser(tmp_path):
    from transformers import TrainerCallback
    return train_grpo.make_pause_callback(TrainerCallback, tmp_path)


def test_stop_file_pauses_at_the_end_of_the_step_and_is_consumed(tmp_path):
    from types import SimpleNamespace
    cb = _pauser(tmp_path)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=7), control)
    assert (control.should_save, control.should_training_stop) == (False, False)
    (tmp_path / train_grpo.STOP_FILE).write_text("")
    cb.on_step_end(None, SimpleNamespace(global_step=8), control)
    assert (control.should_save, control.should_training_stop) == (True, True)
    assert cb.paused_at == 8 and cb.requested == "STOP file"
    assert not (tmp_path / train_grpo.STOP_FILE).exists(), "consumed, so a resume does not re-pause"
    # a later step end does not move the pause step
    cb.on_step_end(None, SimpleNamespace(global_step=9), control)
    assert cb.paused_at == 8


def test_one_signal_pauses_a_second_aborts(tmp_path):
    import signal
    from types import SimpleNamespace
    cb = _pauser(tmp_path)
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    cb.install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        assert cb.requested == "SIGTERM"
        control = SimpleNamespace(should_save=False, should_training_stop=False)
        cb.on_step_end(None, SimpleNamespace(global_step=3), control)
        assert control.should_save and control.should_training_stop and cb.paused_at == 3
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        for s, h in before.items():
            signal.signal(s, h)


def test_pause_exits_before_the_final_evaluation_with_its_own_code():
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert train_grpo.PAUSED_RC == 3
    body = src.split("trainer.train(resume_from_checkpoint=", 1)[1]
    assert body.index("return PAUSED_RC") < body.index("full_val_eval(")
    assert "PAUSED_MARKER).write_text" in body
    # stale markers never survive a new attempt
    assert "for stale in (STOP_FILE, PAUSED_MARKER)" in src


def test_the_driver_and_supervisor_understand_the_pause():
    drivers = REPO / "runs" / "grpo_2026-09-12"
    if not drivers.exists():
        pytest.skip(f"{drivers} not present")
    drv = (drivers / "run_grpo.sh").read_text(encoding="utf-8")
    assert "PAUSED_RC=3" in drv and 'exit $PAUSED_RC' in drv
    sup = (drivers / "supervise.sh").read_text(encoding="utf-8")
    assert '"$OUT/PAUSED"' in sup


def test_qlora_adapters_are_fp32_after_the_trainer_is_built():
    """TRL casts trainable weights to bf16 on a quantized base in its
    constructor; the recipe keeps fp32 adapters on every base."""
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    after = src.split("trainer = GRPOTrainer(", 1)[1]
    assert "for prm in trainer.model.parameters():" in after
    assert "prm.data = prm.data.float()" in after


def test_pause_skips_the_end_of_training_eval(tmp_path):
    """A pause is not the end of training: no off-cadence eval, no best/ move."""
    import train_sft
    from transformers import TrainerCallback
    from types import SimpleNamespace
    HarnessEval, _ = train_sft.make_callbacks(TrainerCallback)
    ev = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)
    seen = []
    ev.run = lambda model, step: seen.append(step)
    cb = train_grpo.make_pause_callback(TrainerCallback, tmp_path,
                                        on_pause=lambda: setattr(ev, "final_eval", False))
    (tmp_path / train_grpo.STOP_FILE).write_text("")
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    cb.on_step_end(None, SimpleNamespace(global_step=13), control)
    ev.on_train_end(None, SimpleNamespace(global_step=13), control, model="m")
    assert seen == [] and cb.paused_at == 13
    ev2 = HarnessEval(None, None, [], tmp_path, 50, 192, 5.0)   # a natural end still evaluates
    ev2.run = lambda model, step: seen.append(step)
    ev2.on_train_end(None, SimpleNamespace(global_step=818), control, model="m")
    assert seen == [818]


def test_adapter_weights_are_reloaded_exactly_after_the_trainer_is_built():
    """TRL's bf16 cast on a quantized base is lossy (9B: 127/250 -> 124/250 at
    step 0); the saved fp32 tensors are copied back in after construction."""
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    after = src.split("trainer = GRPOTrainer(", 1)[1]
    assert "restore_adapter_weights(trainer.model, adapter)" in after
    assert "set_peft_model_state_dict" in src and "adapter_model.safetensors" in src
    assert after.index("restore_adapter_weights(") < after.index("trainer.train(resume_from_checkpoint=")
