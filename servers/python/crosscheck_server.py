#!/usr/bin/env python3
"""crosscheck-agent — Python MCP server.

Exposes four tools to Claude Code over MCP (JSON-RPC 2.0 on stdio):

  confer    — ask multiple LLMs the same question and return their answers
  debate    — bounded round-trip debate; the moderator synthesises the result
  plan      — collaborative planning across LLMs
  review    — have peers review a snippet of code / a proposal

Everything honours the limits in crosscheck.config.json:
  max_rounds, token_cap, max_time_seconds, providers, moderator.

The server is deliberately dependency-light: it uses only the Python stdlib
plus `urllib.request` for HTTP, so `python3 crosscheck_server.py` just works.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "crosscheck.config.json"
CONFIG_EXAMPLE = ROOT / "crosscheck.config.example.json"
ENV_PATH = ROOT / ".env"
TRANSCRIPT_DIR = ROOT / ".crosscheck" / "transcripts"


# ------------------------------------------------------------
# .env + config loading
# ------------------------------------------------------------
def load_env() -> dict[str, str]:
    env: dict[str, str] = dict(os.environ)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env.setdefault(k.strip(), v.strip())
    return env


def load_config() -> dict[str, Any]:
    src = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE
    return json.loads(src.read_text())


ENV = load_env()
CFG = load_config()


# ------------------------------------------------------------
# Provider adapters — all normalised to chat(messages) -> text
# ------------------------------------------------------------
@dataclass
class Provider:
    name: str
    send: Callable[[list[dict], int, float], str]
    model: str


def _http_post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}") from e


def openai_compatible(name: str, url: str, key_env: str, model_env: str, default_model: str) -> Provider | None:
    key = ENV.get(key_env)
    if not key:
        return None
    model = ENV.get(model_env, default_model)

    def send(messages: list[dict], max_tokens: int, temperature: float) -> str:
        resp = _http_post(
            url,
            {"Authorization": f"Bearer {key}"},
            {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=CFG.get("max_time_seconds", 120),
        )
        return resp["choices"][0]["message"]["content"]

    return Provider(name=name, send=send, model=model)


def anthropic_provider() -> Provider | None:
    key = ENV.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    model = ENV.get("ANTHROPIC_MODEL", "claude-opus-4-5")

    def send(messages: list[dict], max_tokens: int, temperature: float) -> str:
        # Anthropic needs system separated and role=assistant/user alternation.
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        convo = [m for m in messages if m["role"] != "system"]
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system:
            body["system"] = system
        resp = _http_post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            body,
            timeout=CFG.get("max_time_seconds", 120),
        )
        return "".join(block.get("text", "") for block in resp.get("content", []))

    return Provider(name="anthropic", send=send, model=model)


def gemini_provider() -> Provider | None:
    key = ENV.get("GEMINI_API_KEY")
    if not key:
        return None
    model = ENV.get("GEMINI_MODEL", "gemini-2.5-pro")

    def send(messages: list[dict], max_tokens: int, temperature: float) -> str:
        contents = []
        system = None
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        body: dict = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        resp = _http_post(url, {}, body, timeout=CFG.get("max_time_seconds", 120))
        cands = resp.get("candidates", [])
        if not cands:
            return ""
        return "".join(p.get("text", "") for p in cands[0]["content"]["parts"])

    return Provider(name="gemini", send=send, model=model)


def build_providers() -> dict[str, Provider]:
    registry: dict[str, Provider | None] = {
        "anthropic": anthropic_provider(),
        "openai":    openai_compatible("openai",   "https://api.openai.com/v1/chat/completions",            "OPENAI_API_KEY",   "OPENAI_MODEL",   "gpt-5"),
        "xai":       openai_compatible("xai",      "https://api.x.ai/v1/chat/completions",                   "XAI_API_KEY",      "XAI_MODEL",      "grok-4-latest"),
        "mistral":   openai_compatible("mistral",  "https://api.mistral.ai/v1/chat/completions",             "MISTRAL_API_KEY",  "MISTRAL_MODEL",  "mistral-large-latest"),
        "groq":      openai_compatible("groq",     "https://api.groq.com/openai/v1/chat/completions",        "GROQ_API_KEY",     "GROQ_MODEL",     "llama-3.3-70b-versatile"),
        "deepseek":  openai_compatible("deepseek", "https://api.deepseek.com/v1/chat/completions",           "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
        "gemini":    gemini_provider(),
    }
    return {k: v for k, v in registry.items() if v is not None}


ALL_PROVIDERS = build_providers()


def active_providers() -> list[Provider]:
    return [ALL_PROVIDERS[p] for p in CFG.get("providers", []) if p in ALL_PROVIDERS]


# ------------------------------------------------------------
# Transcripts
# ------------------------------------------------------------
def write_transcript(kind: str, payload: dict) -> str | None:
    if not CFG.get("log_transcripts", True):
        return None
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = TRANSCRIPT_DIR / f"{stamp}-{kind}.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path.relative_to(ROOT))


# ------------------------------------------------------------
# Tool implementations
# ------------------------------------------------------------
def _per_call_tokens() -> int:
    rounds = max(1, int(CFG.get("max_rounds", 3)))
    providers = max(1, len(active_providers()))
    # Spread the total token budget across rounds * providers.
    return max(256, int(CFG.get("token_cap", 8000)) // (rounds * providers))


def _deadline() -> float:
    return time.monotonic() + float(CFG.get("max_time_seconds", 120))


def _time_left(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _ask_one(p: Provider, messages: list[dict], deadline: float) -> dict:
    budget = _per_call_tokens()
    temp = float(CFG.get("temperature", 0.4))
    if _time_left(deadline) <= 0:
        return {"provider": p.name, "model": p.model, "error": "time budget exhausted"}
    try:
        out = p.send(messages, budget, temp)
        return {"provider": p.name, "model": p.model, "response": out}
    except Exception as e:
        return {"provider": p.name, "model": p.model, "error": str(e)}


def tool_confer(args: dict) -> dict:
    question: str = args["question"]
    context: str = args.get("context", "")
    providers = args.get("providers") or [p.name for p in active_providers()]
    selected = [ALL_PROVIDERS[n] for n in providers if n in ALL_PROVIDERS]
    if not selected:
        return {"error": "no active providers have API keys in .env"}

    system = (
        "You are part of a panel of LLMs consulted by an engineer working inside "
        "Claude Code. Answer directly, cite assumptions, and keep it crisp."
    )
    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "user", "content": f"CONTEXT:\n{context}"})
    messages.append({"role": "user", "content": question})

    deadline = _deadline()
    answers = [_ask_one(p, messages, deadline) for p in selected]
    result = {"tool": "confer", "question": question, "answers": answers}
    result["transcript"] = write_transcript("confer", result)
    return result


def tool_debate(args: dict) -> dict:
    topic: str = args["topic"]
    context: str = args.get("context", "")
    providers = args.get("providers") or [p.name for p in active_providers()]
    selected = [ALL_PROVIDERS[n] for n in providers if n in ALL_PROVIDERS]
    if len(selected) < 2:
        return {"error": "debate needs at least 2 active providers"}

    max_rounds = int(args.get("max_rounds", CFG.get("max_rounds", 3)))
    deadline = _deadline()
    transcript: list[dict] = []
    shared_context = context

    for rnd in range(1, max_rounds + 1):
        if _time_left(deadline) <= 1:
            break
        round_messages = [
            {"role": "system", "content": (
                "You are debating peers from other model families. Round "
                f"{rnd}/{max_rounds}. Disagree where warranted, concede where "
                "right, and keep replies short and specific."
            )},
        ]
        if shared_context:
            round_messages.append({"role": "user", "content": f"CONTEXT:\n{shared_context}"})
        round_messages.append({"role": "user", "content": f"TOPIC: {topic}"})
        if transcript:
            prior = "\n\n".join(
                f"[{e['provider']} — round {e['round']}]\n{e.get('response','(error)')}"
                for e in transcript
            )
            round_messages.append({"role": "user", "content": f"PRIOR TURNS:\n{prior}"})

        for p in selected:
            if _time_left(deadline) <= 1:
                break
            entry = _ask_one(p, round_messages, deadline)
            entry["round"] = rnd
            transcript.append(entry)

    # Moderator synthesises.
    moderator_name = args.get("moderator") or CFG.get("moderator", "anthropic")
    moderator = ALL_PROVIDERS.get(moderator_name) or (selected[0] if selected else None)
    synthesis = None
    if moderator and _time_left(deadline) > 1:
        condensed = "\n\n".join(
            f"[{e['provider']} — round {e['round']}]\n{e.get('response','(error)')}"
            for e in transcript
        )
        synth_messages = [
            {"role": "system", "content": "You are the moderator. Synthesise the debate into a single grounded recommendation."},
            {"role": "user", "content": f"TOPIC: {topic}\n\nTRANSCRIPT:\n{condensed}"},
        ]
        synthesis = _ask_one(moderator, synth_messages, deadline)

    result = {
        "tool": "debate",
        "topic": topic,
        "rounds_completed": max((e["round"] for e in transcript), default=0),
        "transcript": transcript,
        "synthesis": synthesis,
    }
    result["transcript_path"] = write_transcript("debate", result)
    return result


def tool_plan(args: dict) -> dict:
    goal = args["goal"]
    constraints = args.get("constraints", "")
    merged = (
        f"We need a step-by-step plan to achieve this goal.\n\n"
        f"GOAL: {goal}\n\n"
        f"CONSTRAINTS: {constraints or '(none stated)'}\n\n"
        "Return: (1) the plan as numbered steps, (2) risks, (3) alternatives considered."
    )
    return tool_debate({"topic": merged, "context": args.get("context", "")})


def tool_review(args: dict) -> dict:
    snippet = args["snippet"]
    intent = args.get("intent", "")
    question = (
        "Review the following code/proposal as peers. Call out bugs, smells, "
        "missed edge cases, and suggest concrete changes.\n\n"
        f"INTENT: {intent or '(not stated)'}\n\n"
        f"SNIPPET:\n```\n{snippet}\n```"
    )
    return tool_confer({"question": question})


# ------------------------------------------------------------
# MCP server (JSON-RPC 2.0 over stdio)
# ------------------------------------------------------------
TOOLS = {
    "confer": {
        "description": "Ask multiple LLMs the same question and return all answers in parallel.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question or prompt."},
                "context":  {"type": "string", "description": "Optional shared context."},
                "providers": {"type": "array", "items": {"type": "string"}, "description": "Subset of active providers; defaults to all active."},
            },
            "required": ["question"],
        },
        "handler": tool_confer,
    },
    "debate": {
        "description": "Run a bounded multi-round debate across LLMs; moderator synthesises the result.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic":     {"type": "string"},
                "context":   {"type": "string"},
                "providers": {"type": "array", "items": {"type": "string"}},
                "max_rounds":{"type": "integer"},
                "moderator": {"type": "string"},
            },
            "required": ["topic"],
        },
        "handler": tool_debate,
    },
    "plan": {
        "description": "Collaborative planning across LLMs with risks + alternatives.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal":        {"type": "string"},
                "constraints": {"type": "string"},
                "context":     {"type": "string"},
            },
            "required": ["goal"],
        },
        "handler": tool_plan,
    },
    "review": {
        "description": "Peer-review a code snippet or proposal across LLMs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "snippet": {"type": "string"},
                "intent":  {"type": "string"},
            },
            "required": ["snippet"],
        },
        "handler": tool_review,
    },
}


def rpc_result(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def rpc_error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(req: dict) -> dict | None:
    method = req.get("method")
    params = req.get("params") or {}
    id_ = req.get("id")

    if method == "initialize":
        return rpc_result(id_, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "crosscheck-agent", "version": "0.1.0"},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return rpc_result(id_, {
            "tools": [
                {"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
                for n, t in TOOLS.items()
            ]
        })
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        tool = TOOLS.get(name)
        if tool is None:
            return rpc_error(id_, -32601, f"unknown tool: {name}")
        try:
            out = tool["handler"](args)
            return rpc_result(id_, {"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
        except Exception as e:
            return rpc_error(id_, -32000, str(e))
    if id_ is None:
        return None  # unknown notification, ignore
    return rpc_error(id_, -32601, f"unknown method: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
