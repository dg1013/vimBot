"""The box vim runs in (owner decision 2026-11-04, DECISIONS.md).

The design assumes vim is already compromised. The screen in src/scope.py, -Z
and 'nowrite' all sit in or before vim, so one memory bug in vim gets past all
of them at once. Only limits the kernel enforces survive that, so this module
is the boundary and those three are defence in depth.

The box starts empty and adds only what vim needs, instead of starting with the
whole machine and removing things. A path that is not mounted does not exist for
the process inside, so this also covers problems nobody has found yet.

One call is a chain of four programs. They never talk to each other: each sets
something on its own process and replaces itself with the next, and Linux
carries those settings across.

    prlimit      resource limits; they cover everything after it
      nsenter    joins the shared empty network and IPC namespaces
        bwrap    builds the empty root, drops privileges, loads the filter
          vim

Each program starts the next only if its own step worked, so vim cannot start
without its restrictions. A box that fails to start raises SandboxError. It is
never scored as an invalid command: rc 1 alone cannot tell the two apart, and a
reward loop would silently learn from the difference.

scripts/sandbox_containment/README.md has the reasoning and the measurements.

Stdlib only, like the rest of src/.
"""

import atexit
import math
import os
import platform
import re
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import NamedTuple, Optional

# Where the scratch directory appears inside the box. A fixed path keeps every
# path vim sees the same from run to run.
WORK = "/work"

# The whole environment vim gets, so that nothing about the caller reaches the
# oracle. COLUMNS, DISPLAY, API tokens and WSL_INTEROP all stop here.
ENV = {
    "PATH": "/usr/bin",
    "TERM": "dumb",
    "HOME": WORK,
    "TMPDIR": WORK,
    "LC_ALL": "C.UTF-8",
}

# Limits. The floors are far above any gold buffer (the largest is under 1 KB)
# and stop a short command from filling memory or disk: `%t$` repeated 20 times
# returned a valid 1 GiB buffer before these existed. The multiples keep a large
# real file workable under the §1 sample rule.
MIN_FILE_BYTES = 16 * 2**20
MIN_MEMORY_BYTES = 512 * 2**20
OPEN_FILES = 64
EXCEPTION_BYTES = 64 * 2**10

# A box that fails to start is retried, then raised. vim never ran, so retrying
# cannot repeat an edit. The first attempt has no pause.
_RETRY_PAUSES = (0.0, 0.05, 0.5)
_HELPER_START_SECONDS = 10.0

_TOOLS = ("bwrap", "nsenter", "prlimit", "unshare", "sh")


class SandboxError(RuntimeError):
    """The box could not be started. This says nothing about the command."""


class SandboxUnavailable(SandboxError):
    """This machine cannot provide the box at all."""


class Outcome(NamedTuple):
    return_code: Optional[int]
    timed_out: bool


# ------------------------------------------------------------ syscall filter
#
# An allowlist, traced from vim across the corpus and from base-model commands,
# which reach paths gold rows never do. Anything else is refused rather than
# killed: the containment is the same, and a kill raises SIGSYS, which gave
# false "invalid" verdicts and, through WSL's crash handler, wrote process
# memory to the Windows drive.
#
# Numbers are the x86-64 ABI, fixed by the kernel. tests/test_box.py checks them
# against the system header when it is installed.
ALLOWED_SYSCALLS = {
    "read": 0, "write": 1, "close": 3, "lseek": 8, "mmap": 9, "mprotect": 10,
    "munmap": 11, "brk": 12, "rt_sigaction": 13, "rt_sigprocmask": 14,
    "rt_sigreturn": 15, "ioctl": 16, "pread64": 17, "access": 21, "mremap": 25,
    "dup": 32, "getpid": 39, "execve": 59, "exit": 60, "wait4": 61, "uname": 63,
    "fcntl": 72, "fsync": 74, "ftruncate": 77, "getcwd": 79, "chdir": 80,
    "fchdir": 81, "rename": 82, "mkdir": 83, "rmdir": 84, "unlink": 87,
    "readlink": 89, "fchmod": 91, "umask": 95, "sysinfo": 99, "getuid": 102,
    "getgid": 104, "geteuid": 107, "getegid": 108, "sigaltstack": 131,
    "statfs": 137, "arch_prctl": 158, "setxattr": 188, "getxattr": 191,
    "futex": 202, "getdents64": 217, "set_tid_address": 218,
    "restart_syscall": 219, "clock_nanosleep": 230, "exit_group": 231,
    "openat": 257, "newfstatat": 262, "pselect6": 270, "set_robust_list": 273,
    "prlimit64": 302, "getrandom": 318, "rseq": 334,
}

# glibc probes nscd at startup, so these fail softly instead of with EPERM.
SOFT_REFUSED_SYSCALLS = {"socket": 41, "connect": 42}

_LD_ABS = 0x20
_JEQ = 0x15
_JGE = 0x35
_RET = 0x06
_RET_ALLOW = 0x7FFF0000
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000
_EPERM = 1
_EACCES = 13
_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_SYSCALL_BIT = 0x40000000


def compile_filter() -> bytes:
    """Classic BPF for seccomp: 8 bytes per instruction, as bwrap reads it."""
    program = bytearray()

    def add(code, jump_true, jump_false, value):
        program.extend(struct.pack("<HBBI", code, jump_true, jump_false, value))

    # Another architecture or the x32 ABI would renumber every syscall below,
    # which is the classic way past a filter like this one.
    add(_LD_ABS, 0, 0, 4)                    # seccomp_data.arch
    add(_JEQ, 1, 0, _AUDIT_ARCH_X86_64)
    add(_RET, 0, 0, _RET_KILL_PROCESS)
    add(_LD_ABS, 0, 0, 0)                    # seccomp_data.nr
    add(_JGE, 0, 1, _X32_SYSCALL_BIT)
    add(_RET, 0, 0, _RET_KILL_PROCESS)
    for number in sorted(SOFT_REFUSED_SYSCALLS.values()):
        add(_JEQ, 0, 1, number)
        add(_RET, 0, 0, _RET_ERRNO | _EACCES)
    for number in sorted(ALLOWED_SYSCALLS.values()):
        add(_JEQ, 0, 1, number)
        add(_RET, 0, 0, _RET_ALLOW)
    add(_RET, 0, 0, _RET_ERRNO | _EPERM)
    return bytes(program)


FILTER = compile_filter()


# --------------------------------------------------------- the shared helper
#
# Network and IPC namespaces are not created per call. IPC namespaces free at
# about 8/s on this kernel against about 450/s created, so per-call ones hit the
# per-user limit (187,835) after about 7 minutes at full rate and then fail with
# ENOSPC. Short benchmarks cannot see this. Instead one process parks in an empty
# pair and every box joins it. bwrap 0.6.1 can create a network namespace but
# cannot join one, which is why nsenter is in the chain at all.

class _SharedNamespaces:
    """One empty network + IPC namespace, held for this process's lifetime."""

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None

    def pid(self) -> int:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._start()
            return self._proc.pid

    def _start(self) -> subprocess.Popen:
        tools = _tools()
        # `read` returns when stdin reaches EOF, which happens however this
        # process ends, so the helper cannot outlive it.
        proc = subprocess.Popen(
            [tools["unshare"], "--user", "--map-root-user", "--net", "--ipc",
             tools["sh"], "-c", "echo up; read _"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=ENV, cwd="/", start_new_session=True,
        )
        # poll, not select: select cannot watch a descriptor numbered 1024 or
        # above, and a long training process easily holds that many.
        watcher = select.poll()
        watcher.register(proc.stdout, select.POLLIN | select.POLLHUP)
        ready = watcher.poll(_HELPER_START_SECONDS * 1000)
        if not ready or proc.stdout.readline().strip() != b"up":
            proc.kill()
            _, err = proc.communicate()
            raise SandboxError(
                "could not create the shared namespaces: "
                + err.decode("utf-8", "replace").strip()
            )
        proc.stdout.close()
        proc.stderr.close()
        return proc

    def stop(self):
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.stdin.close()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


_HELPER = _SharedNamespaces()
_PROBE: Optional[bool] = None
_PROBE_REASON = ""
# Without this lock a thread pool races the probe (the §9 reward workers use
# about 8). Found 2026-08-18 under a parallel gate.
_PROBE_LOCK = threading.Lock()
_TOOL_PATHS: dict = {}


def _stop_helper():
    _HELPER.stop()


def _after_fork_in_child():
    # A forked child gets its own helper: the parent's Popen is not its child,
    # and a lock held by another thread at fork would stay held forever.
    global _HELPER, _PROBE_LOCK
    _HELPER = _SharedNamespaces()
    _PROBE_LOCK = threading.Lock()


atexit.register(_stop_helper)
os.register_at_fork(after_in_child=_after_fork_in_child)


# ---------------------------------------------------------- what the box needs

def _tools() -> dict:
    if _TOOL_PATHS:
        return _TOOL_PATHS
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise SandboxUnavailable(
            "the box needs x86-64 Linux; the syscall filter is written for "
            f"that ABI and this is {sys.platform}/{platform.machine()}"
        )
    found = {name: shutil.which(name) for name in _TOOLS}
    missing = sorted(name for name, path in found.items() if path is None)
    if missing:
        hint = " (sudo apt install bubblewrap)" if "bwrap" in missing else ""
        raise SandboxUnavailable(
            f"vim will not run without the box; missing {', '.join(missing)}{hint}"
        )
    _TOOL_PATHS.update(found)
    return _TOOL_PATHS


def resolve_vim(vim_bin: str) -> str:
    """The real path of `vim_bin`, which has to exist inside the box."""
    found = shutil.which(vim_bin)
    if found is None:
        raise FileNotFoundError(f"vim binary not found: {vim_bin}")
    real = os.path.realpath(found)
    if not real.startswith("/usr/"):
        raise SandboxUnavailable(
            f"{real} is outside /usr, the only program directory in the box"
        )
    return real


def isolation_available(vim_path: str) -> bool:
    """Probe once whether the box actually holds, using a real vim.

    Empirical on purpose: bwrap existing proves nothing, and the per-call
    status check only proves vim started. -Z is left off so that the kernel
    walls are what is being tested.
    """
    global _PROBE, _PROBE_REASON
    if _PROBE is not None:
        return _PROBE
    with _PROBE_LOCK:
        if _PROBE is not None:
            return _PROBE
        top, work = make_scratch()
        outside = tempfile.mkdtemp(prefix="vimBot_probe_")
        result = False
        try:
            secret = os.path.join(outside, "secret")
            with open(secret, "w") as fh:
                fh.write("x\n")
            steps = (
                f"call writefile(['x'], '{WORK}/inside')",
                f"call writefile(['x'], '{outside}/written')",
                f"if filereadable('{secret}') | "
                f"call writefile(['x'], '{WORK}/leaked') | endif",
                "qa!",
            )
            argv = [vim_path, "-N", "-n", "-u", "NONE", "-i", "NONE",
                    "--not-a-term", "-es"]
            for step in steps:
                argv += ["-c", step]
            try:
                run(argv, top, timeout=10.0)
                result = (os.path.exists(os.path.join(work, "inside"))
                          and not os.path.exists(os.path.join(outside, "written"))
                          and not os.path.exists(os.path.join(work, "leaked")))
                _PROBE_REASON = "" if result else (
                    "the probe reached something outside the box")
            except SandboxError as exc:
                _PROBE_REASON = str(exc)
        finally:
            remove_scratch(top)
            shutil.rmtree(outside, ignore_errors=True)
        _PROBE = result
        return result


def require(vim_path: str) -> None:
    """Raise unless the box is available. vim never runs without it."""
    _tools()
    if not isolation_available(vim_path):
        raise SandboxUnavailable(f"the box is not holding: {_PROBE_REASON}")


# --------------------------------------------------------------- scratch space

def make_scratch() -> tuple:
    """Create (top, work). Only `work` is visible inside the box."""
    top = tempfile.mkdtemp(prefix="vimBot_")
    work = os.path.join(top, "work")
    os.mkdir(work, 0o700)
    return top, work


def remove_scratch(top: str) -> None:
    shutil.rmtree(top, ignore_errors=True)
    if not os.path.lexists(top):
        return
    # Two things leave it standing: a box still dying after a timeout kill, and
    # a directory left with no permissions.
    time.sleep(0.05)
    for root, dirs, _ in os.walk(top):
        for name in dirs:
            path = os.path.join(root, name)
            if not os.path.islink(path):
                try:
                    os.chmod(path, 0o700)
                except OSError:
                    pass
    shutil.rmtree(top, ignore_errors=True)


def file_limit(buffer_bytes: int) -> int:
    return max(MIN_FILE_BYTES, 4 * buffer_bytes)


def memory_limit(buffer_bytes: int) -> int:
    return max(MIN_MEMORY_BYTES, 16 * buffer_bytes)


# ----------------------------------------------------------- running the chain

_STARTED = re.compile(rb'"exit-code"\s*:\s*\d+')


def _command(argv, work, tools, helper_pid, timeout, file_bytes, memory_bytes,
             status_fd, filter_fd) -> list:
    command = [
        tools["prlimit"],
        # Not 0: a piped core_pattern ignores RLIMIT_CORE unless it is exactly
        # 1, and on WSL that pipe writes process memory to the Windows drive.
        "--core=1",
        f"--fsize={file_bytes}",
        f"--nofile={OPEN_FILES}",
        f"--as={memory_bytes}",
        f"--cpu={math.ceil(timeout) + 1}",
        "--",
        # The user namespace has to come along: it is what grants the right to
        # join the other two.
        tools["nsenter"],
        f"--user=/proc/{helper_pid}/ns/user",
        f"--net=/proc/{helper_pid}/ns/net",
        f"--ipc=/proc/{helper_pid}/ns/ipc",
        "--preserve-credentials",
        "--",
        # Not given --unshare-net or --unshare-ipc, so bwrap stays in the two
        # it inherited from nsenter.
        tools["bwrap"],
        "--unshare-user", "--unshare-pid", "--unshare-uts", "--unshare-cgroup",
        "--uid", "0", "--gid", "0", "--hostname", "box",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--dev-bind", "/dev/null", "/dev/null",
        "--dev-bind", "/dev/urandom", "/dev/urandom",
        "--bind", work, WORK,
        "--chdir", WORK,
        # Options apply in order, so this comes after everything it covers.
        "--remount-ro", "/",
        "--new-session", "--die-with-parent", "--cap-drop", "ALL",
        # bwrap writes the "exit-code" record only once the target has started,
        # and closes this fd inside the box, so vim can neither forge it nor
        # stop it being written.
        "--json-status-fd", str(status_fd),
    ]
    if filter_fd is not None:
        command += ["--seccomp", str(filter_fd)]
    return command + ["--"] + list(argv)


def _launch(argv, top, timeout, file_bytes, memory_bytes, filtered):
    """One attempt. Returns an Outcome, or None if the box did not start."""
    tools = _tools()
    work = os.path.join(top, "work")
    status_r, status_w = os.pipe()
    filter_r = None
    try:
        passed = [status_w]
        if filtered:
            filter_r, filter_w = os.pipe()
            # The program is about 1 KB, far below a pipe's capacity, so this
            # cannot block and nothing is written to disk.
            os.write(filter_w, FILTER)
            os.close(filter_w)
            passed.append(filter_r)
        command = _command(argv, work, tools, _HELPER.pid(), timeout,
                           file_bytes, memory_bytes, status_w, filter_r)
        # stdout is discarded and stderr goes to a file outside the box, so a
        # flood costs the harness nothing. RLIMIT_FSIZE caps that file.
        with open(os.path.join(top, "stderr"), "wb") as err:
            try:
                proc = subprocess.run(
                    command, cwd=top, env=ENV, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=err, timeout=timeout,
                    pass_fds=tuple(passed), start_new_session=True,
                )
            except subprocess.TimeoutExpired:
                # subprocess killed the outer bwrap. --die-with-parent takes the
                # box's PID 1 with it, and that ends every process inside.
                return Outcome(None, True)
        os.close(status_w)
        status_w = None
        os.set_blocking(status_r, False)
        try:
            status = os.read(status_r, 65536)
        except BlockingIOError:
            status = b""
        if _STARTED.search(status) is None:
            return None
        return Outcome(proc.returncode, False)
    finally:
        for fd in (status_r, status_w, filter_r):
            if fd is not None:
                os.close(fd)


def run(argv, top, timeout, file_bytes=MIN_FILE_BYTES,
        memory_bytes=MIN_MEMORY_BYTES, filtered=True) -> Outcome:
    """Run `argv` in a fresh box over `top`/work.

    Raises SandboxError if the box does not start, after retrying. `filtered`
    exists only so tests can show the walls hold without the filter; a test pins
    that nothing outside tests/ turns it off.
    """
    failure = ""
    for pause in _RETRY_PAUSES:
        time.sleep(pause)
        try:
            outcome = _launch(argv, top, timeout, file_bytes, memory_bytes, filtered)
        except SandboxError as exc:
            failure = str(exc)
            continue
        if outcome is not None:
            return outcome
        failure = tail(os.path.join(top, "stderr"), 500).strip()
    raise SandboxError(f"the box did not start, so vim never ran: {failure}")


# --------------------------------------------------------- reading results back
#
# The harness runs outside the box, where a symlink planted inside would
# resolve for real. On 2026-11-04 one returned a host file as buffer_after with
# valid=True, and a planted FIFO hung run_vim past its timeout, because this read
# comes after the timeout has ended. So nothing here trusts the file: no
# following links, no blocking, regular files only, and a size cap.

def read_back(path: str, cap: int) -> Optional[str]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        return None
    # Text mode with the default newline handling, as the executor has always
    # read buffers: it keeps CRLF rows comparing the way they always have.
    with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def tail(path: str, chars: int) -> str:
    """The last `chars` characters of a diagnostics file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4 * chars))
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-chars:]
