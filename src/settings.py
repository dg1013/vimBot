"""Project settings: a TOML file that overrides built-in defaults.

Precedence, highest first:

    CLI flag  >  configs/settings.toml  >  built-in default

There is deliberately no environment-variable layer: two sources of truth are
easier to reason about than three, and "why is the timeout 10?" has exactly
one place to look besides the command line. Adding an env layer later would
slot in between CLI and file without breaking anything here.

The defaults below are the single source of truth -- the shipped
configs/settings.toml has every key commented out, so an uncommented line in
that file always means "I am deliberately overriding something."

Unknown sections, unknown keys, and wrong types are hard errors rather than
warnings: a typo like `timout = 10` should fail loudly at startup, not sit
there silently doing nothing.

Stdlib only (tomllib, Python 3.11+).
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = REPO / "configs" / "settings.toml"

# Built-in defaults. Structure defines what the file is allowed to contain:
# any section or key not present here is rejected.
DEFAULTS = {
    "executor": {
        "timeout": 5.0,        # seconds of wall clock per vim run
        "vim_bin": "vim",      # binary to invoke
        "mode": "ex",          # default command mode
        "on_exists": "refuse",  # output_path conflict policy
    },
    "generate": {
        "max_buffer_lines": 60,
        "max_command_len": 200,
        "teacher": "claude-fable-5-in-session",
    },
    "splits": {
        "test_min": 300,
        "test_max": 500,
        "val_frac": 0.10,
        "review_min": 150,
        "seed": 13,
    },
    "stats": {
        "samples": 10,
        "seed": 13,
    },
}

# Value constraints beyond type. Kept here rather than imported from the
# modules that use them, so settings.py has no imports from src/ and cannot
# create a cycle; tests assert these stay in sync with the real definitions.
_ALLOWED = {
    ("executor", "mode"): ("ex", "keys"),
    ("executor", "on_exists"): ("refuse", "overwrite", "suffix"),
}


def _check_value(section: str, key: str, value, default):
    """Validate one value against its default's type and any constraint."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"[{section}] {key}: expected true/false, "
                             f"got {value!r}")
    elif isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"[{section}] {key}: expected a number, "
                             f"got {value!r}")
        value = float(value)
    elif isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"[{section}] {key}: expected an integer, "
                             f"got {value!r}")
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"[{section}] {key}: expected a string, "
                             f"got {value!r}")

    allowed = _ALLOWED.get((section, key))
    if allowed is not None and value not in allowed:
        raise ValueError(f"[{section}] {key}: expected one of {allowed}, "
                         f"got {value!r}")

    if key in ("timeout", "max_buffer_lines", "max_command_len", "samples") \
            and value <= 0:
        raise ValueError(f"[{section}] {key}: must be positive, got {value!r}")
    if key == "val_frac" and not 0.0 <= value <= 1.0:
        raise ValueError(f"[{section}] {key}: must be between 0 and 1, "
                         f"got {value!r}")
    return value


class Settings:
    """Resolved settings, plus where each value came from."""

    def __init__(self, values: dict, sources: dict, path=None):
        self._values = values
        self._sources = sources
        self.path = path

    def __getitem__(self, section: str) -> dict:
        return self._values[section]

    def source(self, section: str, key: str) -> str:
        """'default' or 'file' -- which layer supplied this value."""
        return self._sources[section][key]

    def as_dict(self) -> dict:
        return {s: dict(v) for s, v in self._values.items()}

    def describe(self) -> str:
        """Human-readable dump of every effective value and its origin."""
        origin = self.path or "(no settings file; built-in defaults)"
        lines = [f"settings file: {origin}", ""]
        for section in sorted(self._values):
            lines.append(f"[{section}]")
            for key in sorted(self._values[section]):
                value = self._values[section][key]
                src = self._sources[section][key]
                lines.append(f"  {key:<18} = {value!r:<32} ({src})")
            lines.append("")
        return "\n".join(lines).rstrip()


def load_settings(path=None) -> Settings:
    """Load settings, falling back to built-in defaults.

    `path` None means the standard location (configs/settings.toml). A
    missing file at the standard location is fine -- the repo works without
    one. A missing file at an EXPLICIT path is an error, since the caller
    clearly meant to use it.
    """
    explicit = path is not None
    target = Path(path) if explicit else DEFAULT_SETTINGS_PATH

    values = {s: dict(keys) for s, keys in DEFAULTS.items()}
    sources = {s: {k: "default" for k in keys} for s, keys in DEFAULTS.items()}

    if not target.exists():
        if explicit:
            raise FileNotFoundError(f"settings file not found: {target}")
        return Settings(values, sources, path=None)

    with open(target, "rb") as fh:
        try:
            loaded = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{target}: invalid TOML ({exc})") from None

    for section, keys in loaded.items():
        if section not in DEFAULTS:
            raise ValueError(
                f"{target}: unknown section [{section}] "
                f"(expected one of {sorted(DEFAULTS)})"
            )
        if not isinstance(keys, dict):
            raise ValueError(f"{target}: [{section}] must be a table")
        for key, value in keys.items():
            if key not in DEFAULTS[section]:
                raise ValueError(
                    f"{target}: unknown key {key!r} in [{section}] "
                    f"(expected one of {sorted(DEFAULTS[section])})"
                )
            values[section][key] = _check_value(
                section, key, value, DEFAULTS[section][key]
            )
            sources[section][key] = "file"

    return Settings(values, sources, path=str(target))
