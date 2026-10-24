"""Tests for scripts/train_grpo_gated.py (difficulty-gated GRPO, DECISIONS.md
2026-09-14) -- the gate itself, and the guard that the copy has not drifted
from scripts/train_grpo.py in anything the gate did not mean to change.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import train_grpo  # noqa: E402
import train_grpo_gated as gated  # noqa: E402
from difficulty import ChunkPlanner, DifficultyBook  # noqa: E402

EOS = 7


def _assistant(text):
    return [{"role": "assistant", "content": text}]


class _State:
    def __init__(self, step):
        self.global_step = step


# ------------------------------------------------------------ drift guard

def test_reward_table_and_constants_are_the_originals():
    for name in ("REWARD_STATES", "REWARD_MATCH", "REWARD_VALID_WRONG", "BREVITY_BONUS",
                 "TIMEOUT_RETRIES", "CHECKPOINT_COMPLETE", "STOP_FILE", "PAUSED_MARKER", "PAUSED_RC"):
        assert getattr(gated, name) == getattr(train_grpo, name), name
    for state in train_grpo.REWARD_STATES:
        for tokens in (0, 12, 96, 192):
            assert gated.reward_for(state, tokens, 192) == train_grpo.reward_for(state, tokens, 192)


def test_shared_cli_defaults_are_the_originals():
    """Every option of the original keeps its default in the copy; the copy
    only ADDS the gate's options."""
    orig = {a.dest: a.default for a in train_grpo.build_arg_parser()._actions}
    copy = {a.dest: a.default for a in gated.build_arg_parser()._actions}
    assert set(orig) <= set(copy)
    for dest, default in orig.items():
        assert copy[dest] == default, dest
    added = set(copy) - set(orig)
    assert added == {"difficulty", "pool", "anchor_frac", "cliff_frac", "unprobed",
                     "refresh_every", "difficulty_log", "no_gate"}


def test_classify_and_completion_length_are_the_originals():
    for ids in ([1, 2, 3, EOS], [1, 2, 3], []):
        assert gated.completion_length(ids, EOS) == train_grpo.completion_length(ids, EOS)
    for text in ("ex: %s/a/b/", "keys: dw", "nope", "<think>x</think>\nex: %d"):
        assert gated.classify(text, [1, 2, EOS], EOS) == train_grpo.classify(text, [1, 2, EOS], EOS)


def test_reward_without_a_book_behaves_exactly_like_the_original(monkeypatch):
    fake = lambda case, mode, command, timeout: "match" if command == "%s/a/b/" else "invalid"  # noqa: E731
    monkeypatch.setattr(train_grpo, "execution_state", fake)
    monkeypatch.setattr(gated, "execution_state", fake)
    args = dict(prompts=[None] * 3,
                completions=[_assistant("ex: %s/a/b/"), _assistant("ex: %s/x/"), _assistant("nope")],
                completion_ids=[[1, 2, EOS], [1, 2, EOS], [1, EOS]],
                id=["a", "a", "a"], buffer_before=["a\n"] * 3, buffer_after=["b\n"] * 3, cursor=[[1, 1]] * 3)
    assert gated.make_reward(EOS, 192, 5.0, 2)(**args) == train_grpo.make_reward(EOS, 192, 5.0, 2)(**args)


def test_no_model_ids_and_no_thinking_here():
    src = (REPO / "scripts" / "train_grpo_gated.py").read_text(encoding="utf-8")
    assert "Qwen/" not in src and "hf_id: " not in src
    assert 'reg["hf_id"]' in src
    assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src


def test_the_copy_declares_itself_and_marks_its_additions():
    src = (REPO / "scripts" / "train_grpo_gated.py").read_text(encoding="utf-8")
    assert "A copy of\n`scripts/train_grpo.py`" in src
    assert src.count("[gated]") >= 8
    assert "The original's docstring follows unchanged." in src


# ------------------------------------------------------------ observations

def test_record_observations_groups_a_batch_by_prompt_and_stamps_the_step():
    book = DifficultyBook(["a", "b"])
    ids = ["a", "a", "a", "b", "b", "b"]
    states = ["match", "match", "invalid", "valid_wrong", "valid_wrong", "parse"]
    jobs = {("a", "ex", "x"): [0, 1], ("a", "ex", "y"): [2], ("b", "ex", "z"): [3, 4]}
    rows = gated.record_observations(book, ids, states, jobs, trainer_state=_State(41))
    assert rows == [
        {"id": "a", "k": 2, "G": 3, "step": 42, "source": "train",
         "states": {"match": 2, "invalid": 1}, "distinct": 2},
        {"id": "b", "k": 0, "G": 3, "step": 42, "source": "train",
         "states": {"valid_wrong": 2, "parse": 1}, "distinct": 1}]
    assert book.shelf("a") == "frontier" and book.shelf("b") == "cliff"
    assert gated.record_observations(book, ["a"], ["match"], {}, trainer_state=None)[0]["step"] == 0


def test_reward_with_a_book_records_every_group_and_logs_the_shelves(monkeypatch, tmp_path):
    monkeypatch.setattr(gated, "execution_state",
                        lambda case, mode, command, timeout: "match" if mode == "keys" else "invalid")
    book = DifficultyBook(["a", "b", "c"], pool="latest")
    book.attach_log(tmp_path / "difficulty_log.jsonl")
    fn = gated.make_reward(EOS, len_cap=192, timeout=5.0, workers=2, book=book)
    completions = [_assistant("keys: dw"), _assistant("keys: dw"), _assistant("ex: %d"), _assistant("nope")]
    ids = [[1, 2, EOS], [1, 2, EOS], [1, 2, EOS], [1, EOS]]
    logged = {}
    rewards = fn(prompts=[None] * 4, completions=completions, completion_ids=ids,
                 id=["a", "a", "b", "b"], buffer_before=["x\n"] * 4, buffer_after=["y\n"] * 4,
                 cursor=[[1, 1]] * 4, trainer_state=_State(4),
                 log_metric=lambda k, v: logged.__setitem__(k, v))
    assert rewards[0] > 1.0 and rewards[2:] == [0.0, 0.0]
    assert book.pooled("a") == (2, 2, 1) and book.pooled("b") == (0, 2, 1)
    assert book.shelf("a") == "known" and book.shelf("b") == "cliff" and book.shelf("c") == "unprobed"
    assert logged["gate/known"] == pytest.approx(1 / 3)
    assert logged["gate/cliff"] == pytest.approx(1 / 3)
    assert logged["gate/unprobed"] == pytest.approx(1 / 3)
    assert logged["gate/frontier"] == 0.0
    # the original's metrics are still there, unchanged
    assert logged["reward_state/keys_share"] == pytest.approx(0.5)
    assert logged["reward_state/zero_signal_groups"] == pytest.approx(1.0)
    lines = (tmp_path / "difficulty_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and '"step": 5' in lines[0]


# ----------------------------------------------------------------- sampler

def _pools(frontier, known=(), cliff=()):
    return {"frontier": list(frontier), "known": list(known), "cliff": list(cliff)}


def test_gated_sampler_emits_repeat_samplers_structure_with_the_planners_prompts():
    """Chunks of `batch_size` distinct prompts, each index G times in a row,
    the chunk repeated `repeat_count` times, `num_samples // batch_size`
    chunks per pass -- exactly what TRL's RepeatSampler yields, so the
    trainer's batching sees no difference."""
    pytest.importorskip("torch")
    GatedSampler = gated.gated_sampler_class()
    planner = ChunkPlanner(_pools(range(6)), batch_size=2, anchor_frac=0, cliff_frac=0, seed=1)
    twin = ChunkPlanner(_pools(range(6)), batch_size=2, anchor_frac=0, cliff_frac=0, seed=1)
    sampler = GatedSampler(list(range(6)), planner, mini_repeat_count=3, batch_size=2, repeat_count=2)
    out = list(sampler)
    assert len(out) == len(sampler) == 3 * 2 * 3 * 2
    expected = []
    for _ in range(3):
        chunk = twin.chunk()
        for _ in range(2):
            for idx in chunk:
                expected += [idx] * 3
    assert out == expected
    # every group of 3 consecutive items shares one prompt, and the two
    # repeats of a chunk are identical
    for i in range(0, len(out), 3):
        assert len(set(out[i:i + 3])) == 1
    assert out[0:6] == out[6:12]


def test_gated_sampler_refuses_a_planner_with_the_wrong_chunk_size():
    pytest.importorskip("torch")
    GatedSampler = gated.gated_sampler_class()
    with pytest.raises(ValueError):
        GatedSampler(list(range(6)), ChunkPlanner(_pools(range(6)), batch_size=3), 2, batch_size=2)


def test_gated_trainer_class_only_swaps_the_sampler():
    pytest.importorskip("torch")

    class Args:
        generation_batch_size = 64
        steps_per_generation = 8

    class FakeTrainer:
        num_generations = 8
        num_iterations = 1
        args = Args()
        train_dataset = list(range(100))

        def _get_train_sampler(self, dataset=None):
            raise AssertionError("must be overridden")

    planner = ChunkPlanner(_pools(range(100)), batch_size=8, seed=0)
    Gated = gated.gated_trainer_class(FakeTrainer, planner)
    sampler = Gated()._get_train_sampler()
    assert type(sampler).__name__ == "GatedSampler"
    assert (sampler.batch_size, sampler.mini_repeat_count, sampler.repeat_count) == (8, 8, 8)
    assert len(sampler) == (100 // 8) * 8 * 8 * 8
    assert Gated.__mro__[1] is FakeTrainer and set(vars(Gated)) - {"__module__", "__doc__", "__qualname__",
                                                                     "__firstlineno__", "__static_attributes__"} == {"_get_train_sampler"}


# ----------------------------------------------------------------- refresh

def test_gate_refresh_rebuilds_the_pools_at_its_cadence_and_reports():
    class Base:
        pass

    book = DifficultyBook(["a", "b", "c", "d"], pool="latest")
    book.record("a", 4, 8, 0)
    book.record("b", 8, 8, 0)
    planner = ChunkPlanner({"frontier": [0], "known": [1], "cliff": []}, batch_size=1, seed=0)
    lines = []
    gate = gated.make_gate_refresh(Base, book, planner, every=50, unprobed="frontier",
                                   log=lambda *a, **k: lines.append(a[0]))
    # not at the cadence: nothing happens
    gate.on_step_end(None, _State(49), object())
    assert gate.history == [] and planner.refreshes == 0
    # the run has meanwhile solved a and found c to be mixed
    book.record("a", 8, 8, 50)
    book.record("c", 3, 8, 50)
    gate.on_step_end(None, _State(50), object())
    assert planner.refreshes == 1
    assert planner.sizes() == {"frontier": 2, "known": 2, "cliff": 0}     # c + unprobed d; a + b
    row = gate.history[0]
    assert row["step"] == 50 and row["shelves"] == {"cliff": 0, "frontier": 1, "known": 2, "unprobed": 1}
    assert row["expected_fuel_frontier"] == pytest.approx(7 * 3 / 8 * 5 / 8)
    assert lines and lines[0].startswith("  [gate] step 50: frontier 1  known 2  cliff 0  unprobed 1")
    off = gated.make_gate_refresh(Base, book, planner, every=0, unprobed="frontier", log=lambda *a, **k: None)
    off.on_step_end(None, _State(100), object())
    assert off.history == []


# --------------------------------------------------------------------- CLI

def test_cli_help_needs_no_torch_and_shows_the_gate():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "train_grpo_gated.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    for flag in ("--model", "--adapter", "--resume", "--num-generations", "--scale-rewards",
                 "--difficulty", "--pool", "--anchor-frac", "--cliff-frac", "--unprobed",
                 "--refresh-every", "--no-gate"):
        assert flag in p.stdout


def test_original_trainer_is_untouched_by_this_feature():
    """The gate lives in the copy; the original must not know about it."""
    src = (REPO / "scripts" / "train_grpo.py").read_text(encoding="utf-8")
    assert "difficulty" not in src and "[gated]" not in src
