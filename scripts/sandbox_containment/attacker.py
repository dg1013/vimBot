"""Stands in for a fully compromised vim inside the box.

Run with the system python so it needs nothing from the repo:

    /usr/bin/python3 -I /work/attacker.py [scratch-dir]

It reads config.json from the scratch directory (default /work), tries each
thing below, and writes True for everything it managed to report.json. It never
opens a real secret: host targets are canaries the caller created, or existence
and access checks only. tests/test_box.py also runs it outside the box, to prove
that a False from inside means the box refused and not that the attacker is broken.
"""

import ctypes
import json
import os
import resource
import socket
import subprocess
import sys

WORK = sys.argv[1] if len(sys.argv) > 1 else "/work"
config = json.load(open(os.path.join(WORK, "config.json")))
report = {}


def attempt(name, action):
    try:
        report[name] = bool(action())
    except BaseException:
        report[name] = False


def exists_any(*paths):
    return any(os.path.lexists(p) for p in paths)


def write_file(path):
    with open(path, "w") as fh:
        fh.write("pwned")
    return True


def write_and_remove(path):
    write_file(path)
    os.unlink(path)
    return True


def connect_tcp():
    with socket.create_connection(("127.0.0.1", config["tcp_port"]), timeout=1):
        return True


def connect_unix(path):
    sock = socket.socket(socket.AF_UNIX)
    sock.settimeout(1)
    sock.connect(path)
    sock.close()
    return True


def route_out():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("1.1.1.1", 53))         # a connected UDP socket sends nothing
    return sock.getsockname()[0] not in ("0.0.0.0", "127.0.0.1")


def make_socket(family):
    socket.socket(family, socket.SOCK_STREAM).close()
    return True


def signal_harness():
    os.kill(config["host_pid"], 0)
    return True


def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return True


def run_program():
    return subprocess.run(["/usr/bin/id", "-u"], capture_output=True, timeout=3).returncode == 0


def run_windows():
    return subprocess.run(["/mnt/c/Windows/System32/whoami.exe"], capture_output=True,
                          timeout=5).returncode == 0


libc = ctypes.CDLL(None, use_errno=True)
MS_BIND, MS_REMOUNT = 4096, 32
CLONE_NEWUSER, CLONE_NEWNS = 0x10000000, 0x20000


def remount():
    return libc.mount(b"none", b"/", None, MS_REMOUNT | MS_BIND, None) == 0


def new_namespace():
    return libc.unshare(CLONE_NEWUSER | CLONE_NEWNS) == 0


def plant(make):
    path = os.path.join(WORK, "planted")
    make(path)
    os.unlink(path)
    return True


def raise_memory_limit():
    unlimited = (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS, unlimited)
    return True


def open_descriptors():
    """Everything open besides stdin, stdout and stderr, found without /proc."""
    found = []
    for fd in range(3, 1024):
        try:
            os.fstat(fd)
            found.append(fd)
        except OSError:
            pass
    return found


home = config["home"]
attempt("read_canary", lambda: open(config["canary_file"]).read())
attempt("write_canary_dir", lambda: write_file(os.path.join(config["canary_dir"], "pwned")))
attempt("sees_home", lambda: os.path.lexists(home))
attempt("sees_repo", lambda: os.path.lexists(config["repo"]))
attempt("sees_windows_drives", lambda: exists_any("/mnt/c", "/mnt/d", "/mnt/e"))
attempt("sees_etc_passwd", lambda: os.path.lexists("/etc/passwd"))
attempt("sees_proc", lambda: os.path.lexists("/proc/self"))
attempt("sees_runtime_sockets", lambda: exists_any("/run", "/tmp/.X11-unix", "/mnt/wslg"))
attempt("sees_gpu_or_kernel_devices", lambda: exists_any("/dev/dxg", "/dev/kmsg", "/dev/fuse"))
attempt("write_dev_shm", lambda: write_and_remove("/dev/shm/vimbot_attacker_probe"))
attempt("write_usr", lambda: write_file("/usr/pwned"))
attempt("write_root", lambda: write_file("/pwned"))
attempt("route_to_internet", route_out)
attempt("connect_host_tcp", connect_tcp)
attempt("connect_host_unix_socket", lambda: connect_unix(config["unix_socket"]))
attempt("signal_harness", signal_harness)
attempt("remount_root_writable", remount)
attempt("raise_memory_limit", raise_memory_limit)
attempt("run_windows_program", run_windows)
attempt("socket_inet", lambda: make_socket(socket.AF_INET))
attempt("socket_unix", lambda: make_socket(socket.AF_UNIX))
attempt("socket_vsock", lambda: make_socket(socket.AF_VSOCK))
attempt("fork", fork)
attempt("run_program", run_program)
attempt("new_namespace", new_namespace)
attempt("memfd", lambda: os.close(os.memfd_create("x")) is None)
attempt("plant_symlink", lambda: plant(lambda p: os.symlink("/etc/hostname", p)))
attempt("plant_fifo", lambda: plant(os.mkfifo))
attempt("write_scratch", lambda: write_file(os.path.join(WORK, "own.txt")))
report["open_descriptors"] = open_descriptors()
report["environment"] = sorted(os.environ)

with open(os.path.join(WORK, "report.json"), "w") as fh:
    json.dump(report, fh)
