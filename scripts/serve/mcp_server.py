#!/usr/bin/env python3
"""vimBot as an MCP server (§10), with the question round trip (2026-10-13).

Tools:

    vimBot_translate(instruction, buffer, cursor_line=1, cursor_col=1)
        -> {"type": "command", "mode", "command", "paste", ...}
           or {"type": "question", "question", "dialogue_id", ...}
    vimBot_reply(dialogue_id, reply)
        -> {"type": "command", ...}: the command after the user's answer

It only produces commands or a question: nothing is ever applied (CLAUDE.md
§1). Its one execution is the sample rule's check, in the testing sandbox
on a copy of the buffer it was sent (§1's 2026-10-13 exception). `paste` is
the command as one line to paste on vim's command line (src/paste.py); show
it beside the raw command so the user still learns the keys.

    python scripts/serve/mcp_server.py --endpoint http://127.0.0.1:8011

The model is served separately (scripts/serve/serve.sh). Logs go to
stderr: stdout is the protocol.
"""
import argparse
from typing import Any
import logging
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

INSTRUCTIONS = (
    "vimBot translates a natural-language editing instruction into one vim command. "
    "Call vimBot_translate with the instruction, the buffer text and the cursor. It returns "
    "either a command, or a clarifying question when its own answers disagree. When it "
    "returns a question, put it to the user and call vimBot_reply with the dialogue_id and "
    "their answer. It never runs a command on the user's file: show the command (and its "
    "`paste` form) for the user to run.")


def build_server(roundtrip):
    """An MCPServer whose two tools call `roundtrip` (a roundtrip.RoundTrip)."""
    from mcp.server import MCPServer
    server = MCPServer(name="vimBot", instructions=INSTRUCTIONS)

    @server.tool(name="vimBot_translate", structured_output=True,
                 description="Translate an editing instruction for the given buffer into a vim "
                             "command, or return a clarifying question to ask the user first.")
    def vimbot_translate(instruction: str, buffer: str, cursor_line: int = 1,
                         cursor_col: int = 1) -> dict[str, Any]:
        return roundtrip.translate(instruction, buffer, (cursor_line, cursor_col))

    @server.tool(name="vimBot_reply", structured_output=True,
                 description="Give the user's answer to vimBot's clarifying question; returns "
                             "the vim command.")
    def vimbot_reply(dialogue_id: str, reply: str) -> dict[str, Any]:
        return roundtrip.reply(dialogue_id, reply)

    return server


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="qwen3.5-4b", help="registry name")
    ap.add_argument("--served-name", default=None, help="vLLM's --served-model-name")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-agree", type=int, default=5,
                    help="ask when the most common result holds at most this many samples")
    ap.add_argument("--settings", default=None, help="settings file (the executor timeout)")
    args = ap.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    import endpoint
    from roundtrip import RoundTrip
    rt = RoundTrip(endpoint.load(args.model, args.endpoint, args.served_name),
                   samples=args.samples, max_agree=args.max_agree, settings_path=args.settings)
    build_server(rt).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
