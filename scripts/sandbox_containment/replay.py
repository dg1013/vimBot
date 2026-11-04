"""Check the box against the frozen verdicts of the old wrapper, under load.

Rerun this after any vim, glibc or kernel upgrade. The syscall filter refuses
what it does not know, and a refused call can change vim's behaviour without a
visible error, so a verdict diff is the only way to see it.

    python scripts/sandbox_containment/replay.py --passes 3

The baseline is 22,909 gold rows plus 40,000 base-model commands, which reach
error paths gold rows never do, through the old wrapper on vim.basic. Use more
total calls than the 187,835 namespace limit when checking a change to how
namespaces are created: per-call IPC namespaces looked fine in every short run.

Exits nonzero on any startup failure or any gold difference.
"""

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from evaluate import _norm                 # noqa: E402
from executor import run_vim               # noqa: E402
from sandbox import SandboxError           # noqa: E402

DEFAULT_BASELINE = REPO / "runs" / "sandbox_containment_2026-11-04" / "baseline.jsonl"


def buffer_sha(text):
    if text is None:
        return None
    joined = "\n".join(_norm(text))
    return hashlib.sha256(joined.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def replay_one(row):
    try:
        r = run_vim(row["buffer_before"], row["command"], mode=row["mode"],
                    cursor=tuple(row["cursor"]))
    except SandboxError as exc:
        return {"failed": str(exc)[:120]}
    return {"valid": bool(r["was_valid"]), "buffer_sha": buffer_sha(r["buffer_after"]),
            "timeout": bool(r["timeout"]), "return_code": r["return_code"]}


def kernel_slab_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("SUnreclaim"):
            return int(line.split()[1]) // 1024
    return None


def compare(rows, results, kind):
    index = [i for i, row in enumerate(rows) if row["kind"] == kind]
    usable = [i for i in index if "failed" not in results[i]]
    valid = [i for i in usable if results[i]["valid"] != rows[i]["valid"]]
    buffers = [i for i in usable if results[i]["buffer_sha"] != rows[i]["buffer_sha"]
               and not rows[i]["timeout"] and not results[i]["timeout"]]
    return {
        "rows": len(index), "valid_differs": valid, "buffer_differs": buffers,
        "old_hang_now_returns": sum(rows[i]["timeout"] and not results[i]["timeout"] for i in usable),
        "hangs_only_now": sum(results[i]["timeout"] and not rows[i]["timeout"] for i in usable),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="write the summary here as JSON")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.baseline)]
    print(f"{len(rows)} rows, {args.passes} pass(es), {args.workers} workers, "
          f"SUnreclaim {kernel_slab_mb()} MB", flush=True)
    summary = {"passes": [], "calls": 0}
    bad = False
    first = None
    for number in range(1, args.passes + 1):
        start = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(replay_one, rows, chunksize=20))
        seconds = time.time() - start
        failures = collections.Counter(r["failed"] for r in results if "failed" in r)
        codes = collections.Counter(r.get("return_code") for r in results if "failed" not in r)
        fingerprint = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16]
        entry = {"pass": number, "calls": len(results), "seconds": round(seconds, 1),
                 "calls_per_second": round(len(results) / seconds),
                 "startup_failures": sum(failures.values()),
                 "timeouts": sum(1 for r in results if r.get("timeout")),
                 "return_codes": {str(k): v for k, v in codes.most_common()},
                 "fingerprint": fingerprint, "sunreclaim_mb": kernel_slab_mb()}
        summary["passes"].append(entry)
        summary["calls"] += len(results)
        print(json.dumps(entry), flush=True)
        for message, count in failures.most_common(3):
            print(f"  {count} × {message}", flush=True)
        bad = bad or bool(failures)
        if first is None:
            first = results
    for kind in ("gold", "sample"):
        c = compare(rows, first, kind)
        print(f"[{kind}] {c['rows']} rows: valid differs {len(c['valid_differs'])}, buffer differs "
              f"{len(c['buffer_differs'])}, old hang now returns {c['old_hang_now_returns']}, "
              f"hangs only now {c['hangs_only_now']}", flush=True)
        for i in (c["valid_differs"] + c["buffer_differs"])[:12]:
            print(f"    {rows[i]['mode']} {rows[i]['command'][:60]!r} old valid={rows[i]['valid']} "
                  f"timeout={rows[i]['timeout']} now {first[i]}", flush=True)
        summary[kind] = {k: (len(v) if isinstance(v, list) else v) for k, v in c.items()}
        if kind == "gold" and (c["valid_differs"] or c["buffer_differs"]):
            bad = True
    summary["fingerprints_identical"] = len({p["fingerprint"] for p in summary["passes"]}) == 1
    print("fingerprints identical across passes:", summary["fingerprints_identical"], flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
    print("FAILED" if bad else "OK", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
