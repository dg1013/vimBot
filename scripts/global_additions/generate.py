#!/usr/bin/env python3
"""Candidate global prompts for follow-up #2 (runs/followups_2026-10-16/preregistration.md).

Why these prompts. On train, the served adapter's leg-2 probe shows that the
global skills it fails are the ones the corpus barely teaches. Of 1,238 global
rows, 1,099 are deletes (`g/../d`, `v/../d`; 3-4% never solved). The rare
skills fail often:
  - substitutes on matching lines: 44 rows, 18% never and 52% only sometimes solved;
  - moves, copies, joins and relative-address actions: about 20 rows together;
  - capture groups: 16 rows, 25% never solved.
The hand-written global rows (realistic edits on code, logs, configs) dominate
the never-solved list. These candidates teach those skills on realistic buffers.

A candidate is a domain buffer plus a skill. The skill reads parameters off the
buffer (a component, a level, a section, a department, a path), writes the gold
command and one of several phrasings, and records why the skill is here.
Parameters vary so the set holds many distinct gold commands, not a few
commands repeated: the data-scaling slope counted distinct problems (whole
leakage components), so repeated commands would not buy what it measured.
Nothing is accepted here; validate.py executes every gold and applies the gates.

    python scripts/global_additions/generate.py --out runs/global_additions_2026-10-16/candidates.jsonl
"""
import argparse
import json
import pathlib
import random
import re
import sys

NAMES = ["ana", "bo", "chen", "dana", "eli", "farah", "gus", "hana", "ivo", "jun", "kai", "lena",
         "mo", "nia", "omar", "pia", "quinn", "rui", "sam", "tara", "uma", "vik", "wen", "yara", "zed"]
COMPONENTS = ["auth", "db", "cache", "api", "sched", "mailer", "billing", "search", "queue", "gateway",
              "worker", "proxy"]
LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
MESSAGES = {
    "INFO": ["started worker", "request served", "token refreshed", "job queued", "config loaded",
             "connection opened", "snapshot written", "health check ok"],
    "WARN": ["slow response", "retrying request", "pool nearly full", "deprecated flag used",
             "clock skew detected", "disk usage high"],
    "ERROR": ["connection refused", "timeout after 30s", "write failed", "bad gateway from upstream",
              "checksum mismatch", "lock not acquired"],
    "DEBUG": ["payload size 512", "cache key built", "entering retry loop", "parsed 14 fields"],
}
PATHS = ["/api/users", "/api/orders", "/login", "/static/app.js", "/api/cart", "/health", "/admin",
         "/api/search", "/logout", "/api/invoices", "/api/items", "/signup"]
METHODS = ["GET", "POST", "PUT", "DELETE"]
STATUSES = [200, 201, 301, 302, 404, 500, 503]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Pune", "Hanoi", "Quito", "Riga"]
DEPTS = ["eng", "ops", "sales", "legal", "design", "support"]
STATES = ["active", "inactive", "pending", "on_hold"]
WORDS = ["alpha", "beacon", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo",
         "juniper", "kestrel", "lumen", "meridian", "nimbus", "onyx", "prism"]
SECTIONS = ["server", "database", "cache", "logging", "mail", "auth", "queue", "storage"]
TASK_VERBS = ["fix", "write", "review", "ship", "test", "update", "migrate"]
TASK_OBJECTS = ["login form", "billing report", "cache layer", "docs page", "search index",
                "mail template", "export job", "rate limiter", "onboarding flow", "audit log"]
SUBJECTS = ["weekly report", "invoice due", "build failed", "meeting moved", "access request",
            "release notes", "outage summary"]
FUNCS = ["load_config", "parse_row", "send_mail", "build_index", "fetch_user", "save_report",
         "merge_rows", "check_quota", "render_page", "close_pool", "sync_cache", "export_csv"]
JS_FUNCS = ["loadCart", "saveDraft", "renderList", "fetchUser", "sendForm", "initApp", "syncPrefs",
            "drawChart"]

WHY = {
    "sub-on-match": "Substitute only on matching lines (`:g/pat/s/../../`). Train has 44 such global rows; "
                    "the served adapter never solves 18% of them and solves 52% only sometimes (leg-2 probe).",
    "sub-on-nonmatch": "Substitute on the lines that do not match (`:v/pat/s/../../`). Train has 8 such rows; "
                       "12% are never solved and 25% only sometimes.",
    "capture": "Capture groups inside a global command. Train has 16 such global rows; 25% are never "
               "solved and 44% only sometimes.",
    "move": "Move matching lines (`:g/pat/m..`). Train has 13 such global rows, one of the rarest actions.",
    "copy": "Copy matching lines (`:g/pat/t..`). Train has 4 such global rows.",
    "join": "Join matching lines with a neighbour (`:g/pat/j`). Train has 5 such rows; 20% are never "
            "solved and 60% only sometimes.",
    "relative": "Act on lines relative to each match (`+1d`, `.,+1d`, `.,/pat/d`). Train has a handful; the "
                "never-solved hand-written rows lean on exactly this (fold a marker into the line above).",
    "realistic-delete": "A global delete over a realistic buffer (logs, configs, diffs, conflicts). Most global "
                        "train rows come from synthetic regex corpora; the hand-written global rows are the "
                        "ones the served adapter never solves.",
    "range": "A global command limited to a line range or a pattern range. This is rare in train.",
}


def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def has(lines, rx):
    r = re.compile(rx)
    return [l for l in lines if r.search(l)]


def vesc(s):
    """A literal for a vim pattern delimited by /."""
    return re.sub(r"([\\/.*$^~\[\]])", r"\\\1", s)


# --------------------------------------------------------------------- domains

def d_applog(rng):
    comps = rng.sample(COMPONENTS, rng.randint(3, 5))
    lines = []
    for i in range(rng.randint(9, 14)):
        lvl = pick(rng, ["INFO", "INFO", "WARN", "ERROR", "DEBUG"])
        t = "2026-03-%02d %02d:%02d:%02d" % (rng.randint(1, 28), rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
        lines.append("%s %-5s %s: %s" % (t, lvl, pick(rng, comps), pick(rng, MESSAGES[lvl])))
    return lines


def d_access(rng):
    paths = rng.sample(PATHS, rng.randint(4, 6))
    lines = []
    for i in range(rng.randint(9, 13)):
        ip = "10.%d.%d.%d" % (rng.randint(0, 9), rng.randint(0, 255), rng.randint(1, 254))
        lines.append('%s - - [14/Mar/2026:09:%02d:%02d] "%s %s HTTP/1.1" %d %d'
                     % (ip, rng.randint(0, 59), rng.randint(0, 59), pick(rng, METHODS), pick(rng, paths),
                        pick(rng, STATUSES + [200, 200]), rng.randint(80, 9000)))
    return lines


def d_python(rng):
    out = ["import os", "import json", ""]
    for f in rng.sample(FUNCS, rng.randint(3, 5)):
        out.append("def %s(%s):" % (f, pick(rng, ["path", "row", "user", "data", "items"])))
        out.extend(rng.sample([
            "    value = compute(%s)" % pick(rng, WORDS),
            "    print(\"%s\", value)" % f,
            "    # TODO: handle %s" % pick(rng, ["errors", "retries", "empty input", "unicode", "timeouts"]),
            "    result = value * %d" % rng.randint(2, 9),
            "    print(result)",
            "    return result",
        ], 4))
        out.append("")
    return out[:-1]


def d_js(rng):
    out = ["import { api } from './api.js';", ""]
    for name in rng.sample(JS_FUNCS, rng.randint(3, 4)):
        out.append("function %s(%s) {" % (name, pick(rng, ["id", "opts", "items", "user"])))
        out.extend(rng.sample([
            "  console.log('%s start');" % name,
            "  const data = api.get('/%s');" % pick(rng, ["cart", "user", "items", "draft", "prefs"]),
            "  // TODO: validate %s" % pick(rng, ["input", "response", "ids", "dates"]),
            "  console.log(data);",
            "  return data;",
        ], 4))
        out += ["}", ""]
    return out[:-1]


def d_ini(rng):
    out = []
    for s in rng.sample(SECTIONS, rng.randint(3, 4)):
        out.append("[%s]" % s)
        for k in rng.sample(["host", "port", "timeout", "user", "level", "retries", "path", "enabled"], 3):
            v = {"host": pick(rng, ["db.local", "10.0.0.%d" % rng.randint(2, 99), "mail.internal"]),
                 "port": str(pick(rng, [5432, 6379, 8080, 25, 443])), "timeout": str(pick(rng, [5, 10, 30])),
                 "user": pick(rng, NAMES), "level": pick(rng, ["info", "debug", "warn"]),
                 "retries": str(rng.randint(1, 5)), "path": "/var/%s" % pick(rng, WORDS),
                 "enabled": pick(rng, ["true", "false"])}[k]
            if rng.random() < 0.35:
                out.append("# %s %s" % (pick(rng, ["old", "default", "legacy"]), k))
            out.append("%s = %s" % (k, v))
        out.append("")
    return out[:-1]


def d_env(rng):
    vals = {"DB_HOST": "db.internal", "DB_PORT": "5432", "API_KEY": "k_%d" % rng.randint(1000, 9999),
            "LOG_LEVEL": pick(rng, ["info", "debug"]), "CACHE_TTL": str(pick(rng, [60, 300, 900])),
            "MAIL_FROM": "noreply@%s.io" % pick(rng, WORDS), "REDIS_URL": "redis://cache:6379",
            "TIMEOUT": str(pick(rng, [10, 30, 60])), "DEBUG": pick(rng, ["0", "1"]),
            "REGION": pick(rng, ["eu-west-1", "us-east-2", "ap-south-1"]), "WORKERS": str(rng.randint(2, 16))}
    out = []
    for k in rng.sample(sorted(vals), 8):
        r = rng.random()
        out.append(("# export %s=%s" if r < 0.25 else "export %s=%s" if r < 0.45 else "%s=%s") % (k, vals[k]))
    return out


def d_csv(rng):
    out = ["id,name,city,dept,status"]
    for i in range(rng.randint(8, 12)):
        out.append("%d,%s,%s,%s,%s" % (100 + 3 * i + rng.randint(0, 2), pick(rng, NAMES).capitalize(), pick(rng, CITIES),
                                       pick(rng, DEPTS), pick(rng, STATES[:3] + ["active"])))
    return out


def d_tasks(rng):
    out = ["# %s sprint" % pick(rng, WORDS).capitalize(), ""]
    for i in range(rng.randint(8, 11)):
        out.append("- [%s] %s %s" % ("x" if rng.random() < 0.4 else " ", pick(rng, TASK_VERBS), pick(rng, TASK_OBJECTS)))
    return out


def d_diff(rng):
    out = []
    for h in range(rng.randint(2, 4)):
        a = rng.randint(3, 120)
        out.append("@@ -%d,4 +%d,4 @@ def %s" % (a, a, pick(rng, ["load", "save", "parse", "render", "flush", "close"])))
        w = pick(rng, WORDS)
        ctx = ["    x = %s()" % w, "    return x", "    log(x)", "    y = x + %d" % rng.randint(1, 9)]
        for c in ctx:
            if rng.random() < 0.35:
                out += ["-" + c, "+" + c.replace("x", "item")]
            else:
                out.append(" " + c)
    return out


def d_conflict(rng):
    out = ["CFLAGS = -O%d" % rng.randint(0, 3)]
    for v in rng.sample(["TARGET", "PREFIX", "CC", "LIBS", "VERSION", "OUTDIR", "LDFLAGS"], rng.randint(2, 3)):
        out += ["<<<<<<< HEAD", "%s = %s" % (v, pick(rng, WORDS)), "=======", "%s = %s" % (v, pick(rng, WORDS)),
                ">>>>>>> %s" % pick(rng, ["feature/perf", "fix/build", "main", "release"]),
                "%s: %s" % (pick(rng, ["all", "install", "check"]), pick(rng, ["build", "test", "lint"]))]
    return out


def d_sql(rng):
    cols = rng.sample(["id", "name", "email", "created_at", "status", "total", "region", "owner"], rng.randint(4, 6))
    out = ["-- %s migration" % pick(rng, WORDS), "SELECT %s," % cols[0]]
    out += ["       %s," % c for c in cols[1:-1]] + ["       %s" % cols[-1]]
    out += ["FROM %s" % pick(rng, ["orders", "users", "invoices", "events"]),
            "-- filter %s" % pick(rng, ["by date", "active only", "old rows"]),
            "WHERE %s = '%s';" % (pick(rng, ["status", "region"]), pick(rng, ["open", "eu", "paid"])),
            "UPDATE %s SET total = %d WHERE id = %d;" % (pick(rng, ["orders", "users"]), rng.randint(1, 99), rng.randint(1, 900))]
    return out


def d_mail(rng):
    out = []
    for s in rng.sample(SUBJECTS, rng.randint(3, 4)):
        out += ["From: %s@%s.io" % (pick(rng, NAMES), pick(rng, WORDS[:6])), "Subject: %s" % s,
                "Date: 2026-03-%02d" % rng.randint(1, 28), ""]
    return out[:-1]


# ---------------------------------------------------------------------- skills
# Each returns a list of (instruction, gold_command, tier, family).

def sk_applog(L, rng):
    out = []
    comps = sorted({l.split()[3].rstrip(":") for l in L})
    levels = sorted({l.split()[2] for l in L})
    for _ in range(3):
        lv, c = pick(rng, levels), pick(rng, comps)
        sel = has(L, r" %s +%s: " % (lv, c))
        if not sel or len(sel) == len(L):
            continue
        p = "%s \\+%s:" % (lv, c)
        act = pick(rng, ["copy", "move", "delete", "keep"])
        if act == "copy":
            out.append((pick(rng, ["Copy the %s lines from %s to the end of the log; keep the originals." % (lv, c),
                                   "Duplicate every %s entry logged by %s at the bottom of the file." % (lv, c)]),
                        "g/%s/t$" % p, "T3", "copy"))
        elif act == "move":
            out.append((pick(rng, ["Move the %s lines from %s to the bottom, keeping their order." % (lv, c),
                                   "Gather every %s entry from the %s component at the end of the log, in order." % (lv, c)]),
                        "g/%s/m$" % p, "T3", "move"))
        elif act == "delete":
            out.append((pick(rng, ["Delete the %s entries from %s." % (lv, c),
                                   "Drop every line where %s logged at %s level." % (c, lv)]),
                        "g/%s/d" % p, "T2", "realistic-delete"))
        else:
            out.append((pick(rng, ["Keep only the %s lines from %s." % (lv, c),
                                   "Show me just %s's %s entries and delete everything else." % (c, lv)]),
                        "v/%s/d" % p, "T3", "realistic-delete"))
    if len(comps) >= 2:
        a, b = rng.sample(comps, 2)
        out.append((pick(rng, ["Keep only the lines from the %s and %s components." % (a, b),
                               "Delete every entry that did not come from %s or %s." % (a, b)]),
                    "v/ \\(%s\\|%s\\):/d" % (a, b), "T3", "realistic-delete"))
    if "ERROR" in levels and len(levels) > 1:
        c = pick(rng, comps)
        if has(L, r" ERROR +%s: " % c) and has(L, r" (INFO|WARN|DEBUG) +%s: " % c):
            out.append((pick(rng, ["On the ERROR lines only, rename the component %s to %s." % (c, c.upper()),
                                   "Write the component name in capitals (%s) on the ERROR lines, and nowhere else." % c.upper()]),
                        "g/ ERROR /s/ %s:/ %s:/" % (c, c.upper()), "T3", "sub-on-match"))
        lvls = [x for x in levels if x != "ERROR"]
        if lvls:
            x = pick(rng, lvls)
            out.append((pick(rng, ["Delete every line logged at %s or ERROR level." % x,
                                   "Remove the %s and ERROR entries." % x]),
                        "g/ \\(%s\\|ERROR\\) /d" % x, "T3", "realistic-delete"))
    c = pick(rng, comps)
    if has(L, " %s: " % c):
        out.append((pick(rng, ["For lines from %s, wrap the log level in square brackets." % c,
                               "Put the level in brackets, like [WARN], on the %s lines." % c]),
                    "g/ %s: /s/ \\(INFO\\|WARN\\|ERROR\\|DEBUG\\)\\( *\\) / [\\1]\\2 /" % c, "T3", "capture"))
        out.append((pick(rng, ["On the %s lines, cut the timestamp down to just the time." % c,
                               "Drop the date from every %s entry, keeping the time." % c]),
                    "g/ %s: /s/^\\S\\+ //" % c, "T3", "sub-on-match"))
    return out


def sk_access(L, rng):
    out = []
    meths = sorted({re.search(r'"(\w+) ', l).group(1) for l in L})
    paths = sorted({re.search(r" (/\S*) HTTP", l).group(1) for l in L})
    codes = sorted({re.search(r'" (\d{3}) ', l).group(1) for l in L})
    for _ in range(3):
        m, p = pick(rng, meths), pick(rng, paths)
        pat = '"%s %s ' % (m, vesc(p))
        sel = has(L, re.escape('"%s %s ' % (m, p)))
        if not sel or len(sel) == len(L):
            continue
        act = pick(rng, ["delete", "keep", "copy", "move", "redact"])
        if act == "delete":
            out.append((pick(rng, ["Remove the %s requests to %s." % (m, p), "Delete every %s %s line." % (m, p)]),
                        "g/%s/d" % pat, "T2", "realistic-delete"))
        elif act == "keep":
            out.append((pick(rng, ["Keep only the %s requests for %s." % (m, p),
                                   "Which lines are %s %s? Keep those and delete the rest." % (m, p)]),
                        "v/%s/d" % pat, "T3", "realistic-delete"))
        elif act == "copy":
            out.append((pick(rng, ["Copy each %s %s request to the bottom of the log." % (m, p),
                                   "Duplicate the %s calls to %s at the end of the file; leave the originals." % (m, p)]),
                        "g/%s/t$" % pat, "T3", "copy"))
        elif act == "move":
            out.append((pick(rng, ["Move the %s %s requests to the end, keeping their order." % (m, p),
                                   "Gather every %s request for %s at the bottom, in order." % (m, p)]),
                        "g/%s/m$" % pat, "T3", "move"))
        else:
            out.append((pick(rng, ["On the %s requests, replace the client IP with 0.0.0.0." % m,
                                   "Anonymise the address at the start of every %s line as 0.0.0.0." % m]),
                        'g/"%s /s/^\\S\\+/0.0.0.0/' % m, "T3", "sub-on-match"))
    if len(codes) >= 2:
        a, b = rng.sample(codes, 2)
        sel = has(L, '" (%s|%s) ' % (a, b))
        if 0 < len(sel) < len(L):
            out.append((pick(rng, ["Keep only the responses with status %s or %s." % (a, b),
                                   "Delete every request that did not end in %s or %s." % (a, b)]),
                        'v/" \\(%s\\|%s\\) /d' % (a, b), "T3", "realistic-delete"))
        c = pick(rng, codes)
        if has(L, '" %s ' % c):
            out.append((pick(rng, ["On the lines that returned %s, mask the path as REDACTED." % c,
                                   "Replace the request path with REDACTED on every %s response." % c]),
                        'g/" %s /s/ \\/\\S* HTTP/ REDACTED HTTP/' % c, "T3", "sub-on-match"))
    return out


def sk_python(L, rng):
    out = []
    funcs = [re.match(r"def (\w+)", l).group(1) for l in L if l.startswith("def ")]
    if has(L, r"^\s*print\("):
        out.append((pick(rng, ["Comment out every print call, keeping its indentation.",
                               "Turn each print( line into a comment without moving it left."]),
                    "g/^\\s*print(/s/^\\(\\s*\\)/\\1# /", "T3", "capture"))
    if has(L, "# TODO"):
        who = pick(rng, NAMES)
        out.append((pick(rng, ["Assign every TODO to %s: change 'TODO:' to 'TODO(%s):'." % (who, who),
                               "Mark the TODO comments as %s's, so each TODO: becomes TODO(%s):." % (who, who)]),
                    "g/# TODO:/s/TODO:/TODO(%s):/" % who, "T3", "sub-on-match"))
    for f in rng.sample(funcs, min(2, len(funcs))):
        act = pick(rng, ["hint", "first", "rename"])
        if act == "hint":
            out.append((pick(rng, ["Add the return type '-> dict' to the %s signature." % f,
                                   "Annotate %s as returning dict: put ' -> dict' before the colon." % f]),
                        "g/^def %s(/s/):$/) -> dict:/" % f, "T3", "sub-on-match"))
        elif act == "first":
            out.append((pick(rng, ["Delete the first line of the %s body." % f,
                                   "Remove the line directly under 'def %s'." % f]),
                        "g/^def %s(/+1d" % f, "T3", "relative"))
        else:
            new = f + "_v2"
            out.append((pick(rng, ["Rename the %s definition to %s (just the def line)." % (f, new),
                                   "On the line that defines %s, change the name to %s." % (f, new)]),
                        "g/^def %s(/s/%s/%s/" % (f, f, new), "T3", "sub-on-match"))
    if len(funcs) >= 2:
        out.append((pick(rng, ["Reduce this module to its function signatures.",
                               "Keep only the def lines."]),
                    "v/^def /d", "T2", "realistic-delete"))
    return out


def sk_js(L, rng):
    out = []
    funcs = [re.match(r"function (\w+)", l).group(1) for l in L if l.startswith("function ")]
    if has(L, r"^\s*console\.log"):
        out.append((pick(rng, ["Comment out the console.log calls, keeping their indentation.",
                               "Silence the logging: put // in front of each console.log, after the leading spaces."]),
                    "g/^\\s*console\\.log/s/^\\(\\s*\\)/\\1\\/\\/ /", "T3", "capture"))
        out.append((pick(rng, ["In the console.log lines, switch the single quotes to double quotes.",
                               "Use double quotes instead of single quotes, but only inside console.log calls."]),
                    "g/console\\.log/s/'/\"/g", "T3", "sub-on-match"))
    for f in rng.sample(funcs, min(2, len(funcs))):
        act = pick(rng, ["export", "first", "async"])
        if act == "export":
            out.append((pick(rng, ["Export %s." % f, "Put 'export ' in front of the %s declaration." % f]),
                        "g/^function %s(/s/^/export /" % f, "T3", "sub-on-match"))
        elif act == "first":
            out.append((pick(rng, ["Delete the first statement of %s." % f,
                                   "Remove the line right below the %s header." % f]),
                        "g/^function %s(/+1d" % f, "T3", "relative"))
        else:
            out.append((pick(rng, ["Make %s async." % f, "Turn the %s declaration into an async function." % f]),
                        "g/^function %s(/s/^/async /" % f, "T3", "sub-on-match"))
    if has(L, "// TODO"):
        out.append((pick(rng, ["Move the TODO comments to the end of the file, in order.",
                               "Gather all the // TODO lines at the bottom, keeping their order."]),
                    "g/\\/\\/ TODO/m$", "T3", "move"))
    return out


def sk_ini(L, rng):
    out = []
    secs = [l[1:-1] for l in L if l.startswith("[")]
    if has(L, "^#"):
        out.append((pick(rng, ["Strip the comments and blank lines from this config.",
                               "Delete every comment line and every empty line."]),
                    "g/^\\(#.*\\)\\=$/d", "T3", "realistic-delete"))
        out.append((pick(rng, ["Remove each comment together with the setting directly below it.",
                               "Every '# ...' line documents the setting under it; delete both lines for each comment."]),
                    "g/^#/.,+1d", "T3", "relative"))
    for s in rng.sample(secs, min(2, len(secs))):
        act = pick(rng, ["drop", "upper", "first"])
        if act == "drop" and s != secs[-1]:
            out.append((pick(rng, ["Delete the [%s] section, through the blank line after it." % s,
                                   "Remove the whole %s section, including its trailing blank line." % s]),
                        "g/^\\[%s\\]$/.,/^$/d" % s, "T3", "relative"))
        elif act == "upper":
            out.append((pick(rng, ["Write the [%s] header in capitals." % s,
                                   "Uppercase the name of the %s section." % s]),
                        "g/^\\[%s\\]$/s/.*/\\U&/" % s, "T3", "sub-on-match"))
        elif s != secs[0]:
            idx = L.index("[%s]" % s)
            if has(L[idx:], "^#"):
                out.append((pick(rng, ["From the [%s] section to the end of the file, delete the comment lines." % s,
                                       "Only in %s and the sections after it, remove the # comments." % s]),
                            "/^\\[%s\\]$/,$g/^#/d" % s, "T3", "range"))
    k = pick(rng, ["host", "port", "timeout", "user", "level", "retries", "path", "enabled"])
    if len(has(L, "^%s = " % k)) >= 2:
        out.append((pick(rng, ["Comment out every %s setting." % k,
                               "Disable all the %s lines by putting '# ' in front of them." % k]),
                    "g/^%s = /s/^/# /" % k, "T3", "sub-on-match"))
    return out


def sk_env(L, rng):
    out = []
    if has(L, "^# export "):
        out.append((pick(rng, ["Uncomment the commented-out export lines.",
                               "Re-enable the exports that were commented out with '# '."]),
                    "g/^# export /s/^# //", "T3", "sub-on-match"))
    plain = [l for l in L if re.match(r"^[A-Z_]+=", l)]
    if plain and len(plain) < len(L):
        out.append((pick(rng, ["Add 'export ' in front of the variables that don't have it yet; leave the comments alone.",
                               "Make every plain NAME=value line an export."]),
                    "g/^[A-Z_]\\+=/s/^/export /", "T3", "sub-on-match"))
        out.append((pick(rng, ["Quote the values on the lines that aren't exported or commented.",
                               "Wrap the value in double quotes on each bare NAME=value line."]),
                    "g/^[A-Z_]\\+=/s/=\\(.*\\)/=\"\\1\"/", "T3", "capture"))
    line_of = {re.search(r"([A-Z_]+)=", l).group(1): l for l in L}
    for k in rng.sample(sorted(line_of), 2):
        if rng.random() < 0.5:
            if line_of[k].startswith("#"):          # already commented: commenting again is not a real edit
                continue
            out.append((pick(rng, ["Comment out %s." % k, "Disable the %s line with a leading '# '." % k]),
                        "g/\\<%s=/s/^/# /" % k, "T3", "sub-on-match"))
        else:
            out.append((pick(rng, ["Move %s to the top of the file." % k, "Put the %s setting first." % k]),
                        "g/\\<%s=/m0" % k, "T3", "move"))
    return out


def sk_csv(L, rng):
    out = []
    rows = L[1:]
    sts = sorted({r.rsplit(",", 1)[1] for r in rows})
    depts = sorted({r.split(",")[3] for r in rows})
    for _ in range(3):
        st, d = pick(rng, sts), pick(rng, depts)
        act = pick(rng, ["delete-st", "move-st", "swap-st", "upper-st", "keep-dept", "delete-dept-st", "tag-dept"])
        if act == "delete-st" and 0 < len(has(rows, ",%s$" % st)) < len(rows):
            out.append((pick(rng, ["Delete the %s rows." % st, "Drop every record whose status is %s." % st]),
                        "g/,%s$/d" % st, "T2", "realistic-delete"))
        elif act == "move-st" and 0 < len(has(rows, ",%s$" % st)) < len(rows):
            out.append((pick(rng, ["Move the %s rows to the end, keeping their order." % st,
                                   "Push every %s record to the bottom of the file in the same order." % st]),
                        "g/,%s$/m$" % st, "T3", "move"))
        elif act == "swap-st" and has(rows, ",%s$" % st):
            out.append((pick(rng, ["On the %s rows, swap the name and city columns." % st,
                                   "For %s records only, put the city before the name." % st]),
                        "g/,%s$/s/^\\([^,]*\\),\\([^,]*\\),\\([^,]*\\)/\\1,\\3,\\2/" % st, "T3", "capture"))
        elif act == "upper-st" and has(rows, ",%s$" % st):
            out.append((pick(rng, ["Uppercase the name in every %s row." % st,
                                   "Write the names of %s records in capitals." % st]),
                        "g/,%s$/s/^\\([^,]*\\),\\([^,]*\\)/\\1,\\U\\2/" % st, "T3", "capture"))
        elif act == "keep-dept" and 0 < len(has(rows, ",%s," % d)) < len(rows):
            out.append((pick(rng, ["Keep the header and the %s rows; delete everything else." % d,
                                   "Only the %s department should remain, plus the header line." % d]),
                        "2,$v/,%s,/d" % d, "T3", "range"))
        elif act == "delete-dept-st" and has(rows, ",%s,%s$" % (d, st)):
            out.append((pick(rng, ["Delete the %s people in %s." % (st, d),
                                   "Remove every %s row whose status is %s." % (d, st)]),
                        "g/,%s,%s$/d" % (d, st), "T2", "realistic-delete"))
        elif act == "tag-dept" and has(rows, ",%s," % d):
            out.append((pick(rng, ["Rename the %s department to %s-team on every row." % (d, d),
                                   "Change %s to %s-team in the dept column." % (d, d)]),
                        "g/,%s,/s/,%s,/,%s-team,/" % (d, d, d), "T3", "sub-on-match"))
    out.append((pick(rng, ["Reverse the data rows but keep the header on top.",
                           "Flip the order of the records below the header."]),
                "2,$g/^/m1", "T3", "move"))
    return out


def sk_tasks(L, rng):
    out = []
    done, open_ = has(L, r"^- \[x\]"), has(L, r"^- \[ \]")
    if done and open_:
        out.append((pick(rng, ["Move the finished tasks to the bottom of the list, in their current order.",
                               "Put every checked item after the unchecked ones, keeping the order within each group."]),
                    "g/^- \\[x\\]/m$", "T3", "move"))
        out.append((pick(rng, ["Strike through the finished tasks: wrap the text after '[x] ' in ~~.",
                               "For each done item, surround its description with ~~ so it renders struck out."]),
                    "g/^- \\[x\\]/s/\\] \\(.*\\)/] ~~\\1~~/", "T3", "capture"))
    for _ in range(2):
        v = pick(rng, TASK_VERBS)
        sel = has(L, r"^- \[.\] %s " % v)
        if not sel or len(sel) == len(has(L, "^- ")):
            continue
        act = pick(rng, ["check", "move", "delete", "owner"])
        if act == "check" and has(L, r"^- \[ \] %s " % v):
            out.append((pick(rng, ["Tick off every '%s' task." % v, "Mark all the %s items as done." % v]),
                        "g/^- \\[ \\] %s /s/\\[ \\]/[x]/" % v, "T3", "sub-on-match"))
        elif act == "move":
            out.append((pick(rng, ["Move the %s tasks to the end of the list, in order." % v,
                                   "Put every '%s' item at the bottom, keeping their order." % v]),
                        "g/^- \\[.\\] %s /m$" % v, "T3", "move"))
        elif act == "delete":
            out.append((pick(rng, ["Delete the %s tasks, done or not." % v,
                                   "Remove every item that starts with '%s'." % v]),
                        "g/^- \\[.\\] %s /d" % v, "T2", "realistic-delete"))
        else:
            who = pick(rng, NAMES)
            out.append((pick(rng, ["Assign the %s tasks to @%s at the end of the line." % (v, who),
                                   "Append ' @%s' to every '%s' item." % (who, v)]),
                        "g/^- \\[.\\] %s /s/$/ @%s/" % (v, who), "T3", "sub-on-match"))
    return out


def sk_diff(L, rng):
    out = []
    if has(L, "^-") and has(L, r"^\+"):
        out.append((pick(rng, ["Drop the removed lines from this diff so it shows the new version with context.",
                               "Delete every line the patch removes (the ones starting with -)."]),
                    "g/^-/d", "T2", "realistic-delete"))
        out.append((pick(rng, ["Turn the removed lines into context: replace the leading - with a space.",
                               "Revert the minus markers to spaces on every removed line."]),
                    "g/^-/s/^-/ /", "T3", "sub-on-match"))
        out.append((pick(rng, ["Keep only the hunk headers and the added lines.",
                               "Delete everything except the @@ lines and the + lines."]),
                    "v/^\\(@@\\|+\\)/d", "T3", "realistic-delete"))
    fn = pick(rng, sorted({l.split()[-1] for l in L if l.startswith("@@")}))
    out.append((pick(rng, ["Strip the function names from the hunk headers, ending each header at the closing @@.",
                           "On every @@ line, delete whatever follows the second @@."]),
                "g/^@@/s/ @@ .*/ @@/", "T3", "sub-on-match"))
    out.append((pick(rng, ["Rename the function in the hunk headers from %s to %s_all." % (fn, fn),
                           "On the @@ lines for %s, change the name to %s_all." % (fn, fn)]),
                "g/^@@ .* def %s$/s/%s$/%s_all/" % (fn, fn, fn), "T3", "sub-on-match"))
    return out


def sk_conflict(L, rng):
    out = [
        (pick(rng, ["Resolve every conflict in favour of HEAD: keep the lines above each =======, and remove all the markers.",
                    "Take our side of each merge conflict (the HEAD part) and delete the other side and every marker line."]),
         "%s/^<\\{7}.*\\n//|g/^=\\{7}/.,/^>\\{7}/d", "T3", "relative"),
        (pick(rng, ["Keep both sides of every conflict and just delete the marker lines.",
                    "Remove the <<<<<<<, ======= and >>>>>>> lines but leave both versions in place."]),
         "g/^\\(<\\{7}\\|=\\{7}\\|>\\{7}\\)/d", "T3", "realistic-delete"),
        (pick(rng, ["Take the incoming side of every conflict: drop the HEAD part and all markers.",
                    "Resolve each conflict to the branch version (below =======) and remove the markers."]),
         "%s/^>\\{7}.*\\n//|g/^<\\{7}/.,/^=\\{7}/d", "T3", "relative"),
    ]
    br = sorted({l.split()[1] for l in L if l.startswith(">>>>>>>")})
    b = pick(rng, br)
    out.append((pick(rng, ["Where the incoming branch is %s, rewrite its marker to say 'theirs' instead." % b,
                           "Replace the branch name %s with 'theirs' on the >>>>>>> lines." % b]),
                "g/^>\\{7} %s$/s/%s/theirs/" % (vesc(b), vesc(b)), "T3", "sub-on-match"))
    return out


def sk_sql(L, rng):
    out = [(pick(rng, ["Delete the SQL comments (the lines starting with --).",
                       "Strip the -- comment lines out of this script."]), "g/^--/d", "T2", "realistic-delete"),
           (pick(rng, ["Put the whole column list of the SELECT on one line.",
                       "Join the SELECT's column lines so the list sits on a single line."]),
            "g/,$/.,/[^,]$/j", "T3", "join"),
           (pick(rng, ["Lowercase the SQL keywords SELECT, FROM, WHERE and UPDATE where they start a line.",
                       "Write the leading SELECT/FROM/WHERE/UPDATE keywords in lowercase."]),
            "g/^\\(SELECT\\|FROM\\|WHERE\\|UPDATE\\)\\>/s/^\\w\\+/\\L&/", "T3", "capture")]
    col = pick(rng, [l.strip().rstrip(",") for l in L if re.match(r"^\s+\w+,?$", l)])
    out.append((pick(rng, ["Alias the %s column as %s_value in the select list." % (col, col),
                           "Add 'AS %s_value' after the %s column." % (col, col)]),
                "g/^\\s\\+%s,\\=$/s/%s/%s AS %s_value/" % (col, col, col, col), "T3", "sub-on-match"))
    return out


def sk_mail(L, rng):
    out = [(pick(rng, ["Delete the Date line after every Subject: line.",
                       "Remove the Date line that follows each subject."]), "g/^Subject:/+1d", "T3", "relative"),
           (pick(rng, ["Join each From: line with the Subject: line below it.",
                       "Put each subject on the same line as its sender."]), "g/^From:/j", "T3", "join")]
    subj = pick(rng, [l[len("Subject: "):] for l in L if l.startswith("Subject: ")])
    out.append((pick(rng, ["Prefix the '%s' subject with [urgent]." % subj,
                           "Mark the %s mail as urgent by putting '[urgent] ' before its subject." % subj]),
                "g/^Subject: %s$/s/: /: [urgent] /" % vesc(subj), "T3", "sub-on-match"))
    dom = pick(rng, sorted({l.split("@")[1] for l in L if l.startswith("From: ")}))
    if 0 < len(has(L, "@%s" % re.escape(dom))) < len(has(L, "^From:")):
        out.append((pick(rng, ["Delete each message from %s: its From, Subject and Date lines." % dom,
                               "Remove the three header lines of every mail sent from %s." % dom]),
                    "g/^From: .*@%s$/.,+2d" % vesc(dom), "T3", "relative"))
    return out


def sk_extra(kind, L, rng):
    """Parameterized joins, copies, captures, relative edits and ranges: the
    rarest global skills in train, and the ones the served adapter fails most."""
    out = []
    if kind == "applog":
        c = pick(rng, sorted({l.split()[3].rstrip(":") for l in L}))
        out.append((pick(rng, ["On the %s lines, rewrite the date from YYYY-MM-DD to DD/MM/YYYY." % c,
                               "Change the date format to day/month/year, but only for %s entries." % c]),
                    "g/ %s: /s/^\\(\\d\\+\\)-\\(\\d\\+\\)-\\(\\d\\+\\)/\\3\\/\\2\\/\\1/" % c, "T3", "capture"))
        n = rng.randint(4, 7)
        if has(L[:n], " DEBUG "):
            out.append((pick(rng, ["Among the first %d lines, delete the DEBUG entries." % n,
                                   "Remove DEBUG lines, but only within lines 1 to %d." % n]),
                        "1,%dg/ DEBUG /d" % n, "T3", "range"))
    elif kind == "access":
        m = pick(rng, sorted({re.search(r'"(\w+) ', l).group(1) for l in L}))
        out.append((pick(rng, ["On the %s lines, move the status code to the start of the line." % m,
                               "For every %s request, put its status code first, followed by a space." % m]),
                    'g/"%s /s/^\\(.*\\) \\(\\d\\{3}\\) \\(\\d\\+\\)$/\\2 \\1 \\3/' % m, "T3", "capture"))
        k = rng.randint(4, 6)
        code = pick(rng, sorted({re.search(r'" (\d{3}) ', l).group(1) for l in L[-k:]}))
        out.append((pick(rng, ["In the last %d requests, delete the ones that returned %s." % (k, code),
                               "Remove the %s responses, but only among the final %d lines." % (code, k)]),
                    '$-%d,$g/" %s /d' % (k - 1, code), "T3", "range"))
    elif kind == "python":
        funcs = [re.match(r"def (\w+)", l).group(1) for l in L if l.startswith("def ")]
        f = pick(rng, funcs)
        out.append((pick(rng, ["Give %s an extra '*args' parameter after its existing one." % f,
                               "Add *args to the %s signature, after the current parameter." % f]),
                    "g/^def %s(/s/(\\(\\w\\+\\))/(\\1, *args)/" % f, "T3", "capture"))
        if f != funcs[-1]:
            out.append((pick(rng, ["Delete the %s function, through the blank line after it." % f,
                                   "Remove the whole %s definition and the empty line that follows it." % f]),
                        "g/^def %s(/.,/^$/d" % f, "T3", "relative"))
    elif kind == "js":
        funcs = [re.match(r"function (\w+)", l).group(1) for l in L if l.startswith("function ")]
        f = pick(rng, funcs)
        out.append((pick(rng, ["Give the parameter of %s a default value of {}." % f,
                               "Default %s's argument to an empty object." % f]),
                    "g/^function %s(/s/(\\(\\w\\+\\))/(\\1 = {})/" % f, "T3", "capture"))
        out.append((pick(rng, ["Delete the %s function, from its header to its closing brace." % f,
                               "Remove %s entirely, down to the } that closes it." % f]),
                    "g/^function %s(/.,/^}/d" % f, "T3", "relative"))
    elif kind == "ini":
        secs = [l[1:-1] for l in L if l.startswith("[")]
        s = pick(rng, secs)
        out.append((pick(rng, ["Delete the first setting under [%s]." % s,
                               "Remove the line right below the %s header." % s]),
                    "g/^\\[%s\\]$/+1d" % s, "T3", "relative"))
        if s != secs[-1]:
            out.append((pick(rng, ["Indent the settings of the [%s] section by four spaces." % s,
                                   "Inside %s only, put four spaces in front of every key line." % s]),
                        "/^\\[%s\\]$/,/^$/g/ = /s/^/    /" % s, "T3", "range"))
        keys = sorted({l.split(" = ")[0] for l in L if " = " in l})
        k = pick(rng, keys)
        out.append((pick(rng, ["Quote the value of every %s setting." % k,
                               "Wrap the %s values in double quotes." % k]),
                    "g/^%s = /s/ = \\(.*\\)/ = \"\\1\"/" % k, "T3", "capture"))
        # (joining a comment with its setting would comment the setting out; dropped 2026-10-16)
    elif kind == "env":
        k = pick(rng, sorted({re.search(r"([A-Z_]+)=", l).group(1) for l in L}))
        out.append((pick(rng, ["Quote the value of %s." % k, "Wrap %s's value in double quotes." % k]),
                    "g/\\<%s=/s/=\\(.*\\)/=\"\\1\"/" % k, "T3", "capture"))
    elif kind == "csv":
        rows = L[1:]
        d = pick(rng, sorted({r.split(",")[3] for r in rows}))
        out.append((pick(rng, ["For the %s rows, swap the id and the name." % d,
                               "Put the name before the id on every %s record." % d]),
                    "g/,%s,/s/^\\([^,]*\\),\\([^,]*\\)/\\2,\\1/" % d, "T3", "capture"))
        st = pick(rng, sorted({r.rsplit(",", 1)[1] for r in rows}))
        out.append((pick(rng, ["Copy every %s row to the end of the file." % st,
                               "Duplicate the %s records below the last line." % st]),
                    "g/,%s$/t$" % st, "T3", "copy"))
        if 0 < len(has(rows, ",%s$" % st)) < len(rows):
            out.append((pick(rng, ["Below the header, delete the %s rows." % st,
                                   "From line 2 on, remove every record with status %s." % st]),
                        "2,$g/,%s$/d" % st, "T3", "range"))
    elif kind == "tasks":
        v = pick(rng, TASK_VERBS)
        if has(L, r"^- \[ \] %s " % v):
            out.append((pick(rng, ["Copy the open '%s' tasks to the end of the list." % v,
                                   "Duplicate every unchecked %s item below the last line." % v]),
                        "g/^- \\[ \\] %s /t$" % v, "T3", "copy"))
        if has(L, r"^- \[.\] %s " % v):
            out.append((pick(rng, ["Uppercase the verb of every '%s' task." % v,
                                   "Write '%s' in capitals at the start of each such item." % v]),
                        "g/^- \\[.\\] %s /s/\\] \\(%s\\)/] \\U\\1/" % (v, v), "T3", "capture"))
    elif kind == "sql":
        # (WHERE sits under a -- comment in these buffers, so joining it upward comments it out; dropped)
        out.append((pick(rng, ["Join the FROM line onto the last column of the select list.",
                               "Pull FROM up onto the line before it."]), "g/^FROM /-1j", "T3", "join"))
    elif kind == "mail":
        dom = pick(rng, sorted({l.split("@")[1] for l in L if l.startswith("From: ")}))
        out.append((pick(rng, ["Join each From: line from %s with its Subject: line." % dom,
                               "For mail sent from %s, put the subject on the sender's line." % dom]),
                    "g/^From: .*@%s$/j" % vesc(dom), "T3", "join"))
        subj = pick(rng, [l[len("Subject: "):] for l in L if l.startswith("Subject: ")])
        out.append((pick(rng, ["Remove the sender line above the '%s' subject." % subj,
                               "Delete the From: line of the %s mail." % subj]),
                    "g/^Subject: %s$/-1d" % vesc(subj), "T3", "relative"))
        out.append((pick(rng, ["Reply-mark the '%s' subject: insert 'Re: ' before it." % subj,
                               "Make the %s subject read 'Re: %s'." % (subj, subj)]),
                    "g/^Subject: %s$/s/^Subject: \\(.*\\)/Subject: Re: \\1/" % vesc(subj), "T3", "capture"))
    elif kind == "diff":
        out.append((pick(rng, ["Pair each removed line with its replacement by joining it to the line below.",
                               "Join every - line with the + line that follows it."]), "g/^-/j", "T3", "join"))
    return out


DOMAINS = {"applog": (d_applog, sk_applog), "access": (d_access, sk_access), "python": (d_python, sk_python),
           "js": (d_js, sk_js), "ini": (d_ini, sk_ini), "env": (d_env, sk_env), "csv": (d_csv, sk_csv),
           "tasks": (d_tasks, sk_tasks), "diff": (d_diff, sk_diff), "conflict": (d_conflict, sk_conflict),
           "sql": (d_sql, sk_sql), "mail": (d_mail, sk_mail)}


def generate(n_buffers, seed):
    rng = random.Random(seed)
    out = []
    kinds = sorted(DOMAINS)
    for i in range(n_buffers):
        kind = kinds[i % len(kinds)]
        build, skills = DOMAINS[kind]
        lines = build(rng)
        buf = "\n".join(lines) + "\n"
        for ins, cmd, tier, fam in skills(lines, rng) + sk_extra(kind, lines, rng):
            out.append({"cand": len(out) + 1, "domain": kind, "instruction": ins, "buffer_before": buf,
                        "gold_command": cmd, "mode": "ex", "cursor": [1, 1], "tier": tier,
                        "category": "global", "skill": fam, "why": WHY[fam]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--buffers", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = generate(args.buffers, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d candidates from %d buffers, %d distinct commands -> %s"
          % (len(rows), args.buffers, len({r["gold_command"] for r in rows}), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
