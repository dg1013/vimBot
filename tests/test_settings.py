"""Tests for the settings layer: precedence, validation, and the shipped file.

Precedence under test:  CLI (command-line interface) flag > settings file > built-in default.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from settings import (  # noqa: E402
    DEFAULTS,
    DEFAULT_SETTINGS_PATH,
    Settings,
    load_settings,
)

EXECUTOR = REPO / "src" / "executor.py"


def write_settings(tmp_path, body: str) -> Path:
    path = tmp_path / "settings.toml"
    path.write_text(body)
    return path


# ------------------------------------------------------------------ defaults

def test_settings_object_exposes_values_and_sources():
    s = Settings({k: dict(v) for k, v in DEFAULTS.items()},
                 {k: {kk: "default" for kk in v} for k, v in DEFAULTS.items()})
    assert s["executor"]["timeout"] == 5.0
    assert s["executor"]["mode"] == "ex"
    assert s.source("executor", "timeout") == "default"
    assert s.path is None


# The shipped settings file overrides exactly these, and nothing else. The
# convention is that every key is commented out, so an uncommented line always
# means a deliberate decision -- this dict is where those decisions are
# recorded. Adding an entry here should come with a DECISIONS.md line; a key
# that turns up uncommented WITHOUT an entry is the accident this guards.
SHIPPED_OVERRIDES = {
    # 2026-08-27: the "namespace" default degrades to no OS (operating system) boundary with only
    # a stderr warning, leaving src/scope.py (six known past gaps) as the sole
    # protection. "required" makes that a hard failure instead.
    ("executor", "isolation"): "required",
}


def test_shipped_settings_file_overrides_only_what_is_declared(tmp_path):
    assert DEFAULT_SETTINGS_PATH.exists(), "configs/settings.toml is missing"
    shipped = load_settings()
    bare = load_settings(write_settings(tmp_path, "# empty\n"))

    expected = {s: dict(v) for s, v in DEFAULTS.items()}
    for (section, key), value in SHIPPED_OVERRIDES.items():
        expected[section][key] = value
    assert shipped.as_dict() == expected

    for section in DEFAULTS:
        for key in DEFAULTS[section]:
            want = "file" if (section, key) in SHIPPED_OVERRIDES else "default"
            assert shipped.source(section, key) == want, (
                "configs/settings.toml has an undeclared override at "
                "[%s] %s -- add it to SHIPPED_OVERRIDES with a reason, or "
                "comment the line back out" % (section, key)
            )

    # everything not deliberately overridden still matches a bare load
    for section in DEFAULTS:
        for key in DEFAULTS[section]:
            if (section, key) not in SHIPPED_OVERRIDES:
                assert shipped[section][key] == bare[section][key]


def test_missing_explicit_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.toml")


# ----------------------------------------------------------------- overriding

def test_file_overrides_default_and_records_source(tmp_path):
    p = write_settings(tmp_path, '[executor]\ntimeout = 12.5\n')
    s = load_settings(p)
    assert s["executor"]["timeout"] == 12.5
    assert s.source("executor", "timeout") == "file"
    # untouched keys keep their defaults
    assert s["executor"]["vim_bin"] == "vim"
    assert s.source("executor", "vim_bin") == "default"


def test_int_is_accepted_for_a_float_field(tmp_path):
    s = load_settings(write_settings(tmp_path, "[executor]\ntimeout = 9\n"))
    assert s["executor"]["timeout"] == 9.0
    assert isinstance(s["executor"]["timeout"], float)


def test_every_section_is_overridable(tmp_path):
    p = write_settings(tmp_path, """
[executor]
vim_bin = "/usr/bin/vim.basic"
[generate]
max_buffer_lines = 5
[splits]
seed = 99
[stats]
samples = 3
""")
    s = load_settings(p)
    assert s["executor"]["vim_bin"] == "/usr/bin/vim.basic"
    assert s["generate"]["max_buffer_lines"] == 5
    assert s["splits"]["seed"] == 99
    assert s["stats"]["samples"] == 3


# ----------------------------------------------------------------- validation

@pytest.mark.parametrize("body,fragment", [
    ('[nope]\nx = 1\n', "unknown section"),
    ('[executor]\ntimout = 5\n', "unknown key"),            # typo caught
    ('[executor]\ntimeout = "soon"\n', "expected a number"),
    ('[executor]\nvim_bin = 7\n', "expected a string"),
    ('[generate]\nmax_buffer_lines = 1.5\n', "expected an integer"),
    ('[executor]\nmode = "visual"\n', "expected one of"),
    ('[executor]\ntimeout = -1\n', "must be positive"),
    ('[splits]\nval_frac = 2.0\n', "between 0 and 1"),
])
def test_invalid_settings_are_rejected_loudly(tmp_path, body, fragment):
    with pytest.raises(ValueError) as exc:
        load_settings(write_settings(tmp_path, body))
    assert fragment in str(exc.value)


def test_malformed_toml_is_reported_as_such(tmp_path):
    with pytest.raises(ValueError) as exc:
        load_settings(write_settings(tmp_path, "[executor\ntimeout = 5\n"))
    assert "invalid TOML" in str(exc.value)


def test_allowed_values_match_the_real_definitions():
    # settings.py deliberately does not import from executor.py (no cycle),
    # so this guards against the two drifting apart.
    from settings import _ALLOWED
    assert _ALLOWED[("executor", "mode")] == ("ex", "keys")
    for section, key in _ALLOWED:
        assert key in DEFAULTS[section], (section, key)


# ------------------------------------------------------------ CLI precedence

def test_cli_flag_beats_the_settings_file(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\ntwo\n")
    p = write_settings(tmp_path, '[executor]\nmode = "keys"\n')

    # file says keys; CLI says ex -> CLI wins: `:2d` deletes line 2
    cli_wins = subprocess.run(
        [sys.executable, str(EXECUTOR), "--settings", str(p),
         "--file", str(src), "--command", "2d", "--mode", "ex"],
        capture_output=True, text=True,
    )
    assert cli_wins.returncode == 0
    assert cli_wins.stdout == "one\n"

    # no CLI flag -> the file's "keys" applies: `2d` is an unfinished
    # operator, the Esc-recovery suffix abandons it, nothing changes
    file_applies = subprocess.run(
        [sys.executable, str(EXECUTOR), "--settings", str(p),
         "--file", str(src), "--command", "2d"],
        capture_output=True, text=True,
    )
    assert file_applies.stdout == "one\ntwo\n"
    assert src.read_text() == "one\ntwo\n"


def test_cli_show_settings_reports_sources(tmp_path):
    p = write_settings(tmp_path, "[executor]\ntimeout = 42.0\n")
    proc = subprocess.run(
        [sys.executable, str(EXECUTOR), "--settings", str(p), "--show-settings"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "42.0" in proc.stdout and "(file)" in proc.stdout
    assert "(default)" in proc.stdout


def test_cli_rejects_a_bad_settings_file(tmp_path):
    p = write_settings(tmp_path, "[executor]\ntimeout = -5\n")
    proc = subprocess.run(
        [sys.executable, str(EXECUTOR), "--settings", str(p),
         "--text", "a", "--command", "%s/a/b/"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "must be positive" in (proc.stderr + proc.stdout)
