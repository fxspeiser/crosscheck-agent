#!/usr/bin/env node
// crosscheck-agent — TypeScript MCP server.
// Mirrors the Python reference implementation, using Node's built-in fetch
// and no external dependencies.

import { readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, dirname, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..", "..", "..");
const CONFIG_PATH = resolve(ROOT, "crosscheck.config.json");
const CONFIG_EXAMPLE = resolve(ROOT, "crosscheck.config.example.json");
const ENV_PATH = resolve(ROOT, ".env");
const TRANSCRIPT_DIR = resolve(ROOT, ".crosscheck", "transcripts");

type Msg = { role: "system" | "user" | "assistant"; content: string };
type Config = {
  max_rounds: number;
  token_cap: number;
  max_time_seconds: number;
  providers: string[];
  moderator: string;
  temperature: number;
  log_transcripts: boolean;
};
type Provider = {
  name: string;
  model: string;
  send: (messages: Msg[], maxTokens: number, temperature: number) => Promise<string>;
};

function loadEnv(): Record<string, string> {
  const env: Record<string, string> = { ...process.env as Record<string, string> };
  if (!existsSync(ENV_PATH)) return env;
  for (const raw of readFileSync(ENV_PATH, "utf8").split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const idx = line.indexOf("=");
    const k = line.slice(0, idx).trim();
    const v = line.slice(idx + 1).trim();
    if (!(k in env)) env[k] = v;
  }
  return env;
}

function loadConfig(): Config {
  const src = existsSync(CONFIG_PATH) ? CONFIG_PATH : CONFIG_EXAMPLE;
  return JSON.parse(readFileSync(src, "utf8"));
}

const ENV = loadEnv();
const CFG = loadConfig();

// ------------------------------------------------------------
// Provider adapters
// ------------------------------------------------------------
async function httpPost(url: string, headers: Record<string, string>, body: unknown, timeoutMs: number): Promise<any> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

function openAICompatible(name: string, url: string, keyEnv: string, modelEnv: string, defaultModel: string): Provider | null {
  const key = ENV[keyEnv];
  if (!key) return null;
  const model = ENV[modelEnv] || defaultModel;
  return {
    name, model,
    async send(messages, maxTokens, temperature) {
      const resp = await httpPost(
        url,
        { Authorization: `Bearer ${key}` },
        { model, messages, max_tokens: maxTokens, temperature },
        CFG.max_time_seconds * 1000,
      );
      return resp.choices?.[0]?.message?.content ?? "";
    },
  };
}

function anthropicProvider(): Provider | null {
  const key = ENV.ANTHROPIC_API_KEY;
  if (!key) return null;
  const model = ENV.ANTHROPIC_MODEL || "claude-opus-4-5";
  return {
    name: "anthropic", model,
    async send(messages, maxTokens, temperature) {
      const system = messages.find((m) => m.role === "system")?.content;
      const convo = messages.filter((m) => m.role !== "system");
      const body: Record<string, unknown> = {
        model, max_tokens: maxTokens, temperature, messages: convo,
      };
      if (system) body.system = system;
      const resp = await httpPost(
        "https://api.anthropic.com/v1/messages",
        { "x-api-key": key, "anthropic-version": "2023-06-01" },
        body,
        CFG.max_time_seconds * 1000,
      );
      return (resp.content as Array<{ text?: string }> | undefined)?.map((b) => b.text ?? "").join("") ?? "";
    },
  };
}

function geminiProvider(): Provider | null {
  const key = ENV.GEMINI_API_KEY;
  if (!key) return null;
  const model = ENV.GEMINI_MODEL || "gemini-2.5-pro";
  return {
    name: "gemini", model,
    async send(messages, maxTokens, temperature) {
      const contents: any[] = [];
      let system: string | undefined;
      for (const m of messages) {
        if (m.role === "system") { system = m.content; continue; }
        contents.push({
          role: m.role === "user" ? "user" : "model",
          parts: [{ text: m.content }],
        });
      }
      const body: Record<string, unknown> = {
        contents,
        generationConfig: { maxOutputTokens: maxTokens, temperature },
      };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
      const resp = await httpPost(url, {}, body, CFG.max_time_seconds * 1000);
      const parts = resp.candidates?.[0]?.content?.parts as Array<{ text?: string }> | undefined;
      return parts?.map((p) => p.text ?? "").join("") ?? "";
    },
  };
}

const ALL_PROVIDERS: Record<string, Provider> = Object.fromEntries(
  [
    anthropicProvider(),
    openAICompatible("openai",   "https://api.openai.com/v1/chat/completions",           "OPENAI_API_KEY",   "OPENAI_MODEL",   "gpt-5"),
    openAICompatible("xai",      "https://api.x.ai/v1/chat/completions",                 "XAI_API_KEY",      "XAI_MODEL",      "grok-4-latest"),
    openAICompatible("mistral",  "https://api.mistral.ai/v1/chat/completions",           "MISTRAL_API_KEY",  "MISTRAL_MODEL",  "mistral-large-latest"),
    openAICompatible("groq",     "https://api.groq.com/openai/v1/chat/completions",      "GROQ_API_KEY",     "GROQ_MODEL",     "llama-3.3-70b-versatile"),
    openAICompatible("deepseek", "https://api.deepseek.com/v1/chat/completions",         "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
    geminiProvider(),
  ].filter((p): p is Provider => p !== null).map((p) => [p.name, p]),
);

function activeProviders(): Provider[] {
  return (CFG.providers ?? []).map((n) => ALL_PROVIDERS[n]).filter(Boolean);
}

function writeTranscript(kind: string, payload: unknown): string | null {
  if (!CFG.log_transcripts) return null;
  mkdirSync(TRANSCRIPT_DIR, { recursive: true });
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..+/, "");
  const path = resolve(TRANSCRIPT_DIR, `${stamp}-${kind}.json`);
  writeFileSync(path, JSON.stringify(payload, null, 2));
  return relative(ROOT, path);
}

function perCallTokens(): number {
  const rounds = Math.max(1, CFG.max_rounds);
  const count = Math.max(1, activeProviders().length);
  return Math.max(256, Math.floor(CFG.token_cap / (rounds * count)));
}

async function askOne(p: Provider, messages: Msg[], deadline: number): Promise<Record<string, unknown>> {
  const timeLeft = deadline - Date.now();
  if (timeLeft <= 0) return { provider: p.name, model: p.model, error: "time budget exhausted" };
  try {
    const response = await p.send(messages, perCallTokens(), CFG.temperature);
    return { provider: p.name, model: p.model, response };
  } catch (e) {
    return { provider: p.name, model: p.model, error: (e as Error).message };
  }
}

// ------------------------------------------------------------
// Tools
// ------------------------------------------------------------
async function toolConfer(args: any): Promise<any> {
  const question: string = args.question;
  const context: string = args.context ?? "";
  const names: string[] = args.providers ?? activeProviders().map((p) => p.name);
  const picked = names.map((n) => ALL_PROVIDERS[n]).filter(Boolean);
  if (picked.length === 0) return { error: "no active providers have API keys in .env" };

  const messages: Msg[] = [
    { role: "system", content: "You are part of a panel of LLMs consulted by an engineer inside Claude Code. Answer directly, cite assumptions, stay crisp." },
  ];
  if (context) messages.push({ role: "user", content: `CONTEXT:\n${context}` });
  messages.push({ role: "user", content: question });

  const deadline = Date.now() + CFG.max_time_seconds * 1000;
  const answers = await Promise.all(picked.map((p) => askOne(p, messages, deadline)));
  const result = { tool: "confer", question, answers } as any;
  result.transcript = writeTranscript("confer", result);
  return result;
}

async function toolDebate(args: any): Promise<any> {
  const topic: string = args.topic;
  const context: string = args.context ?? "";
  const names: string[] = args.providers ?? activeProviders().map((p) => p.name);
  const picked = names.map((n) => ALL_PROVIDERS[n]).filter(Boolean);
  if (picked.length < 2) return { error: "debate needs at least 2 active providers" };

  const maxRounds = Number(args.max_rounds ?? CFG.max_rounds);
  const deadline = Date.now() + CFG.max_time_seconds * 1000;
  const transcript: any[] = [];

  for (let rnd = 1; rnd <= maxRounds; rnd++) {
    if (deadline - Date.now() <= 1000) break;
    const roundMessages: Msg[] = [
      { role: "system", content: `You are debating peers from other model families. Round ${rnd}/${maxRounds}. Disagree where warranted, concede where right, keep replies short.` },
    ];
    if (context) roundMessages.push({ role: "user", content: `CONTEXT:\n${context}` });
    roundMessages.push({ role: "user", content: `TOPIC: ${topic}` });
    if (transcript.length > 0) {
      const prior = transcript.map((e) => `[${e.provider} — round ${e.round}]\n${e.response ?? "(error)"}`).join("\n\n");
      roundMessages.push({ role: "user", content: `PRIOR TURNS:\n${prior}` });
    }
    for (const p of picked) {
      if (deadline - Date.now() <= 1000) break;
      const entry = await askOne(p, roundMessages, deadline);
      (entry as any).round = rnd;
      transcript.push(entry);
    }
  }

  const modName = args.moderator ?? CFG.moderator;
  const moderator = ALL_PROVIDERS[modName] ?? picked[0];
  let synthesis: any = null;
  if (moderator && deadline - Date.now() > 1000) {
    const condensed = transcript.map((e) => `[${e.provider} — round ${e.round}]\n${e.response ?? "(error)"}`).join("\n\n");
    synthesis = await askOne(
      moderator,
      [
        { role: "system", content: "You are the moderator. Synthesise the debate into a single grounded recommendation." },
        { role: "user", content: `TOPIC: ${topic}\n\nTRANSCRIPT:\n${condensed}` },
      ],
      deadline,
    );
  }

  const result = {
    tool: "debate",
    topic,
    rounds_completed: transcript.reduce((m, e) => Math.max(m, e.round), 0),
    transcript,
    synthesis,
  } as any;
  result.transcript_path = writeTranscript("debate", result);
  return result;
}

async function toolPlan(args: any): Promise<any> {
  const goal = args.goal;
  const constraints = args.constraints ?? "";
  return toolDebate({
    topic: `We need a step-by-step plan to achieve this goal.\n\nGOAL: ${goal}\n\nCONSTRAINTS: ${constraints || "(none stated)"}\n\nReturn: (1) the plan as numbered steps, (2) risks, (3) alternatives considered.`,
    context: args.context ?? "",
  });
}

async function toolReview(args: any): Promise<any> {
  return toolConfer({
    question: `Review the following code/proposal as peers. Call out bugs, smells, missed edge cases, and suggest concrete changes.\n\nINTENT: ${args.intent || "(not stated)"}\n\nSNIPPET:\n\`\`\`\n${args.snippet}\n\`\`\``,
  });
}

// ------------------------------------------------------------
// MCP JSON-RPC loop
// ------------------------------------------------------------
const TOOLS: Record<string, { description: string; inputSchema: any; handler: (a: any) => Promise<any> }> = {
  confer: {
    description: "Ask multiple LLMs the same question and return all answers in parallel.",
    inputSchema: { type: "object", properties: {
      question: { type: "string" }, context: { type: "string" },
      providers: { type: "array", items: { type: "string" } },
    }, required: ["question"] },
    handler: toolConfer,
  },
  debate: {
    description: "Run a bounded multi-round debate across LLMs; moderator synthesises the result.",
    inputSchema: { type: "object", properties: {
      topic: { type: "string" }, context: { type: "string" },
      providers: { type: "array", items: { type: "string" } },
      max_rounds: { type: "integer" }, moderator: { type: "string" },
    }, required: ["topic"] },
    handler: toolDebate,
  },
  plan: {
    description: "Collaborative planning across LLMs with risks + alternatives.",
    inputSchema: { type: "object", properties: {
      goal: { type: "string" }, constraints: { type: "string" }, context: { type: "string" },
    }, required: ["goal"] },
    handler: toolPlan,
  },
  review: {
    description: "Peer-review a code snippet or proposal across LLMs.",
    inputSchema: { type: "object", properties: {
      snippet: { type: "string" }, intent: { type: "string" },
    }, required: ["snippet"] },
    handler: toolReview,
  },
};

function result(id: any, r: any) { return { jsonrpc: "2.0", id, result: r }; }
function error(id: any, code: number, message: string) { return { jsonrpc: "2.0", id, error: { code, message } }; }

async function handle(req: any): Promise<any | null> {
  const { method, params = {}, id } = req;
  if (method === "initialize") {
    return result(id, {
      protocolVersion: "2024-11-05",
      capabilities: { tools: {} },
      serverInfo: { name: "crosscheck-agent", version: "0.1.0" },
    });
  }
  if (method === "notifications/initialized") return null;
  if (method === "tools/list") {
    return result(id, { tools: Object.entries(TOOLS).map(([name, t]) => ({ name, description: t.description, inputSchema: t.inputSchema })) });
  }
  if (method === "tools/call") {
    const tool = TOOLS[params.name];
    if (!tool) return error(id, -32601, `unknown tool: ${params.name}`);
    try {
      const out = await tool.handler(params.arguments ?? {});
      return result(id, { content: [{ type: "text", text: JSON.stringify(out, null, 2) }] });
    } catch (e) {
      return error(id, -32000, (e as Error).message);
    }
  }
  if (id === undefined || id === null) return null;
  return error(id, -32601, `unknown method: ${method}`);
}

const rl = createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let req: any;
  try { req = JSON.parse(trimmed); } catch { return; }
  const resp = await handle(req);
  if (resp !== null) {
    process.stdout.write(JSON.stringify(resp) + "\n");
  }
});
