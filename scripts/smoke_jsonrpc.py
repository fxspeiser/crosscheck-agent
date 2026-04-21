#!/usr/bin/env python3
"""
Smoke-test an MCP stdio server by running a tiny JSON-RPC handshake.

Usage:
  python3 scripts/smoke_jsonrpc.py <server-cmd> [args...]

Example:
  python3 scripts/smoke_jsonrpc.py python3 servers/python/crosscheck_server.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def _read_responses(proc: subprocess.Popen[str], want_ids: set[int], timeout_s: float) -> dict[int, Any]:
    deadline = time.monotonic() + timeout_s
    out: dict[int, Any] = {}
    assert proc.stdout is not None
    while want_ids - out.keys():
        if time.monotonic() >= deadline:
            _die(f"timeout waiting for responses: missing={sorted(want_ids - out.keys())}")
        line = proc.stdout.readline()
        if line == "":
            _die("server stdout closed unexpectedly")
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(msg, dict):
            continue
        msg_id = msg.get("id")
        if isinstance(msg_id, int) and msg_id in want_ids:
            out[msg_id] = msg
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        _die("usage: smoke_jsonrpc.py <server-cmd> [args...]")

    cmd = argv[1:]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None

    try:
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "list_providers", "arguments": {}},
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        resps = _read_responses(proc, {1, 2, 3}, timeout_s=5.0)

        init = resps[1].get("result") or {}
        proto = init.get("protocolVersion")
        if proto != "2024-11-05":
            _die(f"unexpected protocolVersion: {proto!r}")

        tools_list = (resps[2].get("result") or {}).get("tools")
        if not isinstance(tools_list, list):
            _die("tools/list: result.tools is not a list")
        tool_names = {t.get("name") for t in tools_list if isinstance(t, dict)}
        required = {"list_providers", "confer", "debate", "plan", "review"}
        missing = required - tool_names
        if missing:
            _die(f"tools/list: missing tools: {sorted(missing)}")

        tool_call = resps[3].get("result") or {}
        content = tool_call.get("content")
        if not (isinstance(content, list) and content and isinstance(content[0], dict)):
            _die("tools/call(list_providers): result.content missing")
        text = content[0].get("text")
        if not isinstance(text, str):
            _die("tools/call(list_providers): content[0].text missing")
        payload = json.loads(text)
        if not isinstance(payload, dict) or "providers" not in payload:
            _die("list_providers payload missing 'providers'")
        if not isinstance(payload.get("providers"), list):
            _die("list_providers payload 'providers' is not a list")

        return 0
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

