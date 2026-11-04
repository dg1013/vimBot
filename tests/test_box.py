"""The box (src/sandbox.py): what compromised code inside it can still reach.

The rest of the adversarial battery asks whether a vim *command* can do damage.
These tests assume vim itself is compromised and run native code in its place,
so they do not depend on the screen, -Z or 'nowrite' at all.

Each attack below worked against the old wrapper on 2026-11-04 unless it says
otherwise; scripts/sandbox_containment/README.md has the before and after. Load-
bearing like the rest of the battery: these encode real findings.
"""

import json
import os
import pathlib
import re
import resource
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import sandbox
from executor import VIM_BIN, run_vim, run_vim_on_file
from sandbox import SandboxError, SandboxUnavailable

REPO = pathlib.Path(__file__).resolve().parents[1]
ATTACKER = REPO / "scripts" / "sandbox_containment" / "attacker.py"
PYTHON = "/usr/bin/python3"
SH = "/usr/bin/sh"

needs_python = pytest.mark.skipif(not os.path.exists(PYTHON), reason="no system python")


# ------------------------------------------------- compromised code inside

@pytest.fixture(scope="module", params=[True, False], ids=["filter-on", "filter-off"])
def attack(request, tmp_path_factory):
    """Run the attacker once per filter setting, with canaries on the host."""
    host = tmp_path_factory.mktemp("host")
    canary_dir = host / "canary"
    canary_dir.mkdir()
    canary_file = canary_dir / "secret"
    canary_file.write_text("SECRET")

    tcp = socket.socket()
    tcp.bind(("127.0.0.1", 0))
    tcp.listen(1)
    unix_path = str(host / "s")
    unix = socket.socket(socket.AF_UNIX)
    unix.bind(unix_path)
    unix.listen(1)

    top, work = sandbox.make_scratch()
    shutil.copy(ATTACKER, os.path.join(work, "attacker.py"))
    with open(os.path.join(work, "config.json"), "w") as fh:
        json.dump({
            "canary_file": str(canary_file), "canary_dir": str(canary_dir),
            "home": str(pathlib.Path.home()), "repo": str(REPO),
            "tcp_port": tcp.getsockname()[1], "unix_socket": unix_path,
            "host_pid": os.getpid(),
        }, fh)
    try:
        outcome = sandbox.run([PYTHON, "-I", f"{sandbox.WORK}/attacker.py"], top,
                              timeout=30.0, filtered=request.param)
        path = os.path.join(work, "report.json")
        assert os.path.exists(path), (outcome, sandbox.tail(os.path.join(top, "stderr"), 800))
        with open(path) as fh:
            report = json.load(fh)
    finally:
        sandbox.remove_scratch(top)
        tcp.close()
        unix.close()
    report["_filtered"] = request.param
    report["_canary_untouched"] = sorted(p.name for p in canary_dir.iterdir()) == ["secret"]
    return report


# The walls that must hold with the filter off, so they do not rest on it.
WALLS = [
    "read_canary", "write_canary_dir", "sees_home", "sees_repo", "sees_windows_drives",
    "sees_etc_passwd", "sees_proc", "sees_runtime_sockets", "sees_gpu_or_kernel_devices",
    "write_dev_shm", "write_usr", "write_root", "route_to_internet", "connect_host_tcp",
    "connect_host_unix_socket", "signal_harness", "remount_root_writable",
    "raise_memory_limit", "run_windows_program",
]
# What only the filter closes. A socket family is not namespaced, and AF_VSOCK
# reaches Windows from WSL, so this list is why the filter exists.
FILTER_ONLY = [
    "socket_inet", "socket_unix", "socket_vsock", "fork", "run_program", "new_namespace",
    "memfd", "plant_symlink", "plant_fifo",
]


@needs_python
@pytest.mark.parametrize("name", WALLS)
def test_compromised_code_cannot(attack, name):
    assert attack[name] is False, f"reached {name} from inside the box"


@needs_python
@pytest.mark.parametrize("name", FILTER_ONLY)
def test_the_filter_refuses(attack, name):
    if not attack["_filtered"]:
        pytest.skip("the filter is what closes this")
    assert attack[name] is False, f"{name} got past the syscall filter"


@needs_python
def test_nothing_on_the_host_changed(attack):
    assert attack["_canary_untouched"]


@needs_python
def test_the_scratch_directory_is_all_that_is_left(attack):
    assert attack["write_scratch"] is True


@needs_python
def test_no_descriptor_crosses_into_the_box(attack):
    # The status fd is what startup detection rests on. If vim could reach it,
    # vim could forge "started", and the filter pipe or a harness file would be
    # a path out of the box.
    assert attack["open_descriptors"] == []


@needs_python
def test_the_environment_is_exactly_the_fixed_one(attack):
    # bwrap adds PWD, which is always /work.
    assert set(attack["environment"]) - {"PWD"} == set(sandbox.ENV)


@needs_python
def test_the_attacker_really_can_do_these_things_outside_the_box(tmp_path):
    # Without this, every False above could just mean the attacker is broken.
    canary = tmp_path / "canary"
    canary.mkdir()
    (canary / "secret").write_text("SECRET")
    tcp = socket.socket()
    tcp.bind(("127.0.0.1", 0))
    tcp.listen(1)
    unix_path = str(tmp_path / "s")
    unix = socket.socket(socket.AF_UNIX)
    unix.bind(unix_path)
    unix.listen(1)
    work = tmp_path / "work"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({
        "canary_file": str(canary / "secret"), "canary_dir": str(canary),
        "home": str(pathlib.Path.home()), "repo": str(REPO),
        "tcp_port": tcp.getsockname()[1], "unix_socket": unix_path,
        "host_pid": os.getpid(),
    }))
    try:
        subprocess.run([PYTHON, "-I", str(ATTACKER), str(work)], check=True, timeout=30,
                       capture_output=True)
    finally:
        tcp.close()
        unix.close()
    report = json.loads((work / "report.json").read_text())
    for name in ("read_canary", "write_canary_dir", "sees_home", "sees_repo", "sees_etc_passwd",
                 "sees_proc", "connect_host_tcp", "connect_host_unix_socket", "signal_harness",
                 "socket_inet", "socket_unix", "fork", "run_program", "memfd", "plant_symlink",
                 "plant_fifo", "write_scratch"):
        assert report[name] is True, f"the attacker could not do {name} even outside the box"


# ------------------------------------------------- no escape needed
#
# These passed every layer of the old wrapper with an ordinary command.

def test_clipboard_registers_reach_nothing():
    for command in ("put +", "put *"):
        r = run_vim("seed\n", command)
        assert r["buffer_after"] == "seed\n", command


def test_the_callers_environment_does_not_reach_vim(monkeypatch):
    monkeypatch.setenv("VIMBOT_CANARY_TOKEN", "SECRET")
    monkeypatch.setenv("DISPLAY", ":0")
    r = run_vim("seed\n", "put =$VIMBOT_CANARY_TOKEN")
    assert "SECRET" not in r["buffer_after"]
    r = run_vim("seed\n", "put =$HOME")
    assert r["buffer_after"].split("\n")[1] == sandbox.WORK


def test_a_doubling_command_stops_at_the_memory_limit():
    # 79 characters, passes the screen, and returned a valid 1 GiB buffer.
    command = "|".join(["%t$"] * 20)
    r = run_vim("x" * 1024 + "\n", command, timeout=5)
    assert r["was_valid"] is False
    assert r["timeout"] is False, "the limit should stop it well before the timeout"
    assert "E342" in r["stderr"]
    assert len(r["buffer_after"]) < 2 * 2**20


# ------------------------------------------------- hostile output
#
# The harness reads results outside the box, where a planted link resolves for
# real. The filter stops the planting, so the first two run with it off: the
# reader has to hold by itself.

def _plant_over_the_buffer(body, timeout=5.0):
    top, work = sandbox.make_scratch()
    try:
        path = os.path.join(work, "buffer.txt")
        with open(path, "w") as fh:
            fh.write("seed\n")
        outcome = sandbox.run([SH, "-c", body], top, timeout=timeout, filtered=False)
        return outcome, sandbox.read_back(path, sandbox.MIN_FILE_BYTES)
    finally:
        sandbox.remove_scratch(top)


def test_a_planted_symlink_is_not_followed():
    _, after = _plant_over_the_buffer(
        f"rm -f {sandbox.WORK}/buffer.txt; ln -s /etc/hostname {sandbox.WORK}/buffer.txt")
    assert after is None


def test_a_planted_fifo_does_not_hang_the_read():
    t0 = time.time()
    _, after = _plant_over_the_buffer(
        f"rm -f {sandbox.WORK}/buffer.txt; mkfifo {sandbox.WORK}/buffer.txt")
    assert after is None
    assert time.time() - t0 < 5


def test_read_back_refuses_what_it_should(tmp_path):
    regular = tmp_path / "regular"
    regular.write_text("ok\n")
    assert sandbox.read_back(str(regular), 100) == "ok\n"
    assert sandbox.read_back(str(regular), 2) is None            # over the cap
    assert sandbox.read_back(str(tmp_path), 100) is None         # a directory
    assert sandbox.read_back(str(tmp_path / "missing"), 100) is None


def test_read_back_keeps_the_newline_handling_buffers_have_always_had(tmp_path):
    # Text mode turns CRLF into LF; bytes would not, and rows would compare differently.
    path = tmp_path / "crlf"
    path.write_bytes(b"a\r\nb\n")
    assert sandbox.read_back(str(path), 100) == "a\nb\n"


def test_nothing_outlives_a_timeout():
    tag = f"vimbot-orphan-{os.getpid()}"
    top, _ = sandbox.make_scratch()
    try:
        outcome = sandbox.run([SH, "-c", f"sh -c 'sleep 30; : {tag}' & sleep 30"], top,
                              timeout=1.0, filtered=False)
    finally:
        sandbox.remove_scratch(top)
    assert outcome.timed_out is True
    deadline = time.time() + 2
    alive = True
    while alive and time.time() < deadline:
        listing = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        alive = any(tag in line and "ps -eo" not in line for line in listing.splitlines())
    assert not alive, "a process inside the box survived the timeout"


def test_stdout_flood_costs_the_harness_nothing():
    top, _ = sandbox.make_scratch()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        outcome = sandbox.run([SH, "-c", "exec yes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"], top,
                              timeout=1.0, filtered=False)
    finally:
        sandbox.remove_scratch(top)
    assert outcome.timed_out is True
    grew_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
    assert grew_kib < 100 * 1024


def test_stderr_flood_is_capped_by_the_file_limit():
    top, _ = sandbox.make_scratch()
    try:
        outcome = sandbox.run([SH, "-c", "exec yes xxxxxxxxxxxxxxxx >&2"], top, timeout=5.0,
                              file_bytes=2**20, filtered=False)
        size = os.path.getsize(os.path.join(top, "stderr"))
    finally:
        sandbox.remove_scratch(top)
    assert size <= 2**20
    assert outcome.return_code == 128 + signal.SIGXFSZ


def test_a_directory_with_no_permissions_is_still_removed():
    top, work = sandbox.make_scratch()
    os.mkdir(os.path.join(work, "locked"), 0o000)
    sandbox.remove_scratch(top)
    assert not os.path.lexists(top)


# ------------------------------------------------- limits

def test_a_crash_cannot_reach_a_piped_core_handler():
    # RLIMIT_CORE=0 does not stop a piped core_pattern; on WSL that pipe wrote
    # process memory to C:. Only exactly 1 does.
    argv = sandbox._command(["x"], "/w", dict.fromkeys(sandbox._TOOLS, "t"), 1, 5.0, 1, 1, 7, 8)
    assert "--core=1" in argv
    top, _ = sandbox.make_scratch()
    try:
        outcome = sandbox.run([SH, "-c", "kill -SEGV $$"], top, timeout=5.0, filtered=False)
    finally:
        sandbox.remove_scratch(top)
    assert outcome.return_code == 128 + signal.SIGSEGV


def test_limits_scale_with_a_large_real_file():
    # Small buffers get the floors, so training is unaffected; a real file under
    # the §1 sample rule must not be refused just for being big.
    assert sandbox.file_limit(1000) == sandbox.MIN_FILE_BYTES
    assert sandbox.memory_limit(1000) == sandbox.MIN_MEMORY_BYTES
    size = 40 * 2**20
    assert sandbox.file_limit(size) >= 2 * size
    r = run_vim("line of text\n" * (size // 13), "1d", timeout=30)
    assert r["was_valid"] is True
    assert len(r["buffer_after"]) == (size // 13 - 1) * 13


# ------------------------------------------------- it fails closed

def test_a_box_that_cannot_start_raises_rather_than_scoring_invalid(monkeypatch):
    real = sandbox._command

    def broken(*args):
        return ["/usr/bin/false"] + real(*args)
    monkeypatch.setattr(sandbox, "_command", broken)
    monkeypatch.setattr(sandbox, "_RETRY_PAUSES", (0.0, 0.0))
    with pytest.raises(SandboxError):
        run_vim("seed\n", "s/seed/x/")


def test_a_failing_command_is_a_verdict_and_not_a_sandbox_error():
    r = run_vim("seed\n", "s/nothere/x/")
    assert r["was_valid"] is False and r["return_code"] == 1


def test_vim_cannot_forge_a_startup_failure():
    # Text on stderr must mean nothing to the harness, or a hostile command
    # could abort a whole training run by printing it.
    top, _ = sandbox.make_scratch()
    try:
        outcome = sandbox.run([SH, "-c", "echo 'bwrap: Creating new namespace failed' >&2; exit 1"],
                              top, timeout=5.0)
    finally:
        sandbox.remove_scratch(top)
    assert outcome.return_code == 1


def test_a_missing_tool_refuses_to_run_vim(monkeypatch):
    monkeypatch.setattr(sandbox, "_TOOL_PATHS", {})
    real_which = shutil.which
    monkeypatch.setattr(sandbox.shutil, "which",
                        lambda name, *a, **k: None if name == "bwrap" else real_which(name, *a, **k))
    with pytest.raises(SandboxUnavailable, match="bubblewrap"):
        sandbox.require("/usr/bin/vim.basic")


def test_a_holding_probe_is_required(monkeypatch):
    monkeypatch.setattr(sandbox, "_PROBE", False)
    monkeypatch.setattr(sandbox, "_PROBE_REASON", "test")
    with pytest.raises(SandboxUnavailable):
        run_vim("seed\n", "1")


def test_a_vim_outside_the_box_is_refused(tmp_path):
    fake = tmp_path / "vim"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    with pytest.raises(SandboxUnavailable, match="outside /usr"):
        run_vim("seed\n", "1", vim_bin=str(fake))


def test_a_dead_helper_is_replaced():
    first = sandbox._HELPER.pid()
    os.kill(first, signal.SIGKILL)
    deadline = time.time() + 2
    while sandbox._HELPER._proc.poll() is None and time.time() < deadline:
        time.sleep(0.01)
    assert run_vim("seed\n", "s/seed/x/")["buffer_after"] == "x\n"
    assert sandbox._HELPER.pid() != first


def test_a_helper_does_not_outlive_its_process():
    code = ("import sys; sys.path.insert(0, %r); import sandbox; print(sandbox._HELPER.pid(), flush=True); "
            "import os; os._exit(0)" % str(REPO / "src"))
    pid = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout)
    deadline = time.time() + 2
    while os.path.exists(f"/proc/{pid}") and time.time() < deadline:
        time.sleep(0.01)
    assert not os.path.exists(f"/proc/{pid}")


# ------------------------------------------------- how it is used

def test_the_default_vim_is_the_pinned_build():
    assert VIM_BIN == "vim.basic"
    assert sandbox.resolve_vim(VIM_BIN).endswith("vim.basic")


def test_only_tests_can_turn_the_filter_off():
    offenders = []
    for root in ("src", "scripts"):
        for path in (REPO / root).rglob("*.py"):
            if "filtered=False" in path.read_text(errors="replace"):
                offenders.append(str(path))
    assert offenders == []


def test_a_thread_pool_gets_correct_results():
    # The §9 reward workers.
    def one(i):
        return run_vim(f"value {i}\n", "s/value/seen/")["buffer_after"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, range(200)))
    assert results == [f"seen {i}\n" for i in range(200)]


def test_a_forked_child_can_use_the_box():
    run_vim("seed\n", "s/seed/x/")          # the parent already holds a helper
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_end)
            os.write(write_end, run_vim("seed\n", "s/seed/y/")["buffer_after"].encode())
        finally:
            os._exit(0)
    os.close(write_end)
    os.waitpid(pid, 0)
    with os.fdopen(read_end) as fh:
        assert fh.read() == "y\n"


def test_a_harness_with_many_open_files_still_works():
    # A training process holds far more descriptors than the 64 the box allows,
    # and select() cannot watch one numbered 1024 or above. Both have to work
    # from a harness that is already that deep.
    import subprocess as sp
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    held = [os.open("/dev/null", os.O_RDONLY) for _ in range(1100)]
    code = (
        "import sys; sys.path.insert(0, %r); from executor import run_vim; "
        "print(run_vim('seed\\n', 's/seed/x/')['buffer_after'] == 'x\\n')" % str(REPO / "src")
    )
    try:
        # a fresh process, so the helper has to start from up here
        out = sp.run([sys.executable, "-c", code], capture_output=True, text=True, close_fds=False,
                     timeout=30)
        assert out.stdout.strip() == "True", out.stderr[-500:]
        assert run_vim("seed\n", "s/seed/y/")["buffer_after"] == "y\n"
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_the_buffer_path_is_the_same_on_every_run():
    # It used to be a random temp path, which `"%p` could put in a buffer.
    assert run_vim("seed\n", 'put =expand("%")')["buffer_after"] == run_vim("seed\n", 'put =expand("%")')["buffer_after"]


# ------------------------------------------------- the filter itself

def test_the_filter_table_matches_the_kernel_abi():
    header = None
    for path in ("/usr/include/x86_64-linux-gnu/asm/unistd_64.h", "/usr/include/asm/unistd_64.h"):
        if os.path.exists(path):
            header = pathlib.Path(path).read_text()
    if header is None:
        pytest.skip("no kernel header installed")
    numbers = {m[1]: int(m[2]) for m in re.finditer(r"#define __NR_(\w+)\s+(\d+)", header)}
    for name, number in {**sandbox.ALLOWED_SYSCALLS, **sandbox.SOFT_REFUSED_SYSCALLS}.items():
        assert numbers[name] == number, name


@pytest.mark.parametrize("name", [
    "fork", "vfork", "clone", "clone3", "ptrace", "mount", "umount2", "pivot_root", "chroot",
    "unshare", "setns", "socketpair", "bind", "listen", "accept", "sendto", "bpf", "userfaultfd",
    "io_uring_setup", "keyctl", "add_key", "perf_event_open", "memfd_create", "symlink", "symlinkat",
    "link", "linkat", "mknod", "mknodat", "process_vm_readv", "process_vm_writev", "kill", "tkill",
    "tgkill", "shmget", "msgget", "semget", "init_module", "kexec_load",
])
def test_nothing_dangerous_is_on_the_allowlist(name):
    assert name not in sandbox.ALLOWED_SYSCALLS
    assert name not in sandbox.SOFT_REFUSED_SYSCALLS


def test_the_filter_is_well_formed():
    assert len(sandbox.FILTER) % 8 == 0
    assert len(sandbox.FILTER) / 8 < 4096        # the kernel's limit
    assert len(sandbox.FILTER) < 4096            # well inside one pipe buffer
