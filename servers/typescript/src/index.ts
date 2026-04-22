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

type Msg = { role: "system" | "user" | "assistant"; content: string };
type Config = {
  max_rounds: number;
  token_cap: number;
  max_time_seconds: number;
  providers: string[];
  moderator: string;
  temperature: number;
  log_transcripts: boolean;
  transcript_dir?: string;
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
const TRANSCRIPT_DIR = resolve(ROOT, CFG.transcript_dir ?? ".crosscheck/transcripts");

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

function isOpenAIReasoningModel(model: string): boolean {
  const m = model.toLowerCase();
  return m.startsWith("gpt-5") || m.startsWith("o1") || m.startsWith("o3") || m.startsWith("o4");
}

function openAICompatible(name: string, url: string, keyEnv: string, modelEnv: string, defaultModel: string): Provider | null {
  const key = ENV[keyEnv];
  if (!key) return null;
  const model = ENV[modelEnv] || defaultModel;
  const isOpenAI = url.includes("api.openai.com");
  return {
    name, model,
    async send(messages, maxTokens, temperature) {
      const body: Record<string, unknown> = { model, messages };
      if (isOpenAI && isOpenAIReasoningModel(model)) {
        body.max_completion_tokens = maxTokens;
        // Reasoning models only accept temperature=1 (default); omit.
      } else {
        body.max_tokens = maxTokens;
        body.temperature = temperature;
      }
      const resp = await httpPost(
        url,
        { Authorization: `Bearer ${key}` },
        body,
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
  const stamp = String(Date.now());
  const path = resolve(TRANSCRIPT_DIR, `${stamp}-${kind}.json`);
  writeFileSync(path, JSON.stringify(payload, null, 2));
  return relative(ROOT, path);
}

function perCallTokens(totalCalls: number): number {
  const calls = Math.max(1, Math.floor(totalCalls));
  return Math.max(256, Math.floor(CFG.token_cap / calls));
}

async function askOne(p: Provider, messages: Msg[], deadline: number, maxTokens: number): Promise<Record<string, unknown>> {
  const timeLeft = deadline - Date.now();
  if (timeLeft <= 0) return { provider: p.name, model: p.model, error: "time budget exhausted" };
  try {
    const response = await p.send(messages, maxTokens, CFG.temperature);
    return { provider: p.name, model: p.model, response };
  } catch (e) {
    return { provider: p.name, model: p.model, error: (e as Error).message };
  }
}

const KNOWN_PROVIDERS = ["anthropic", "openai", "xai", "gemini", "mistral", "groq", "deepseek"];

function resolveProviders(names: string[] | undefined | null): { resolved: Provider[]; unknown: string[] } {
  if (!names || names.length === 0) return { resolved: activeProviders(), unknown: [] };
  const seen = new Set<string>();
  const resolved: Provider[] = [];
  const unknown: string[] = [];
  for (const n of names) {
    const key = n.trim().toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    const prov = ALL_PROVIDERS[key];
    if (prov) resolved.push(prov);
    else unknown.push(n);
  }
  return { resolved, unknown };
}

function unknownProviderError(unknown: string[]) {
  const needsKey: string[] = [];
  const typos: string[] = [];
  for (const n of unknown) {
    (KNOWN_PROVIDERS.includes(n.trim().toLowerCase()) ? needsKey : typos).push(n);
  }
  return {
    error: "requested providers are not available",
    unknown,
    needs_api_key_in_env: needsKey,
    unrecognised_names: typos,
    available_now: Object.keys(ALL_PROVIDERS).sort(),
  };
}

// ------------------------------------------------------------
// Tools
// ------------------------------------------------------------
async function toolListProviders(_args: any): Promise<any> {
  const active = new Set(CFG.providers ?? []);
  return {
    providers: KNOWN_PROVIDERS.map((name) => {
      const prov = ALL_PROVIDERS[name];
      return {
        name,
        available: Boolean(prov),
        active: active.has(name),
        model: prov?.model ?? null,
      };
    }),
    moderator_default: CFG.moderator,
    usage_hint:
      "Pass a 'providers' array to confer/debate/plan/review to pick an ad-hoc subset, " +
      "e.g. providers=['openai','gemini']. Omit the field to use the configured active set.",
  };
}

async function toolConfer(args: any): Promise<any> {
  const question: string = args.question;
  const context: string = args.context ?? "";
  const { resolved: picked, unknown } = resolveProviders(args.providers);
  if (unknown.length > 0 && picked.length === 0) return unknownProviderError(unknown);
  if (picked.length === 0) return { error: "no active providers have API keys in .env" };

  const messages: Msg[] = [
    { role: "system", content: "You are part of a panel of LLMs consulted by an engineer inside Claude Code. Answer directly, cite assumptions, stay crisp." },
  ];
  if (context) messages.push({ role: "user", content: `CONTEXT:\n${context}` });
  messages.push({ role: "user", content: question });

  const deadline = Date.now() + CFG.max_time_seconds * 1000;
  const maxTokens = perCallTokens(picked.length);
  const answers = await Promise.all(picked.map((p) => askOne(p, messages, deadline, maxTokens)));
  const result = { tool: "confer", question, answers } as any;
  if (unknown.length > 0) result.skipped_unknown_providers = unknown;
  const path = writeTranscript("confer", result);
  if (path) {
    result.transcript_path = path;
    result.transcript = path; // backwards-compatible alias
  }
  return result;
}

async function toolDebate(args: any): Promise<any> {
  const topic: string = args.topic;
  const context: string = args.context ?? "";
  const { resolved: picked, unknown } = resolveProviders(args.providers);
  if (unknown.length > 0 && picked.length < 2) return unknownProviderError(unknown);
  if (picked.length < 2) return { error: "debate needs at least 2 providers with keys in .env", available_now: Object.keys(ALL_PROVIDERS).sort() };

  const maxRounds = Number(args.max_rounds ?? CFG.max_rounds);
  const deadline = Date.now() + CFG.max_time_seconds * 1000;
  const transcript: any[] = [];
  const maxTokens = perCallTokens(Math.max(1, maxRounds) * picked.length + 1);

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
      const entry = await askOne(p, roundMessages, deadline, maxTokens);
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
      maxTokens,
    );
  }

  const result = {
    tool: "debate",
    topic,
    rounds_completed: transcript.reduce((m, e) => Math.max(m, e.round), 0),
    transcript,
    synthesis,
  } as any;
  if (unknown.length > 0) result.skipped_unknown_providers = unknown;
  result.transcript_path = writeTranscript("debate", result);
  return result;
}

async function toolPlan(args: any): Promise<any> {
  const goal = args.goal;
  const constraints = args.constraints ?? "";
  return toolDebate({
    topic: `We need a step-by-step plan to achieve this goal.\n\nGOAL: ${goal}\n\nCONSTRAINTS: ${constraints || "(none stated)"}\n\nReturn: (1) the plan as numbered steps, (2) risks, (3) alternatives considered.`,
    context: args.context ?? "",
    providers: args.providers,
    moderator: args.moderator,
  });
}

async function toolReview(args: any): Promise<any> {
  return toolConfer({
    question: `Review the following code/proposal as peers. Call out bugs, smells, missed edge cases, and suggest concrete changes.\n\nINTENT: ${args.intent || "(not stated)"}\n\nSNIPPET:\n\`\`\`\n${args.snippet}\n\`\`\``,
    providers: args.providers,
  });
}

// ------------------------------------------------------------
// MCP JSON-RPC loop
// ------------------------------------------------------------
const PROVIDER_DESC = "Ad-hoc subset of provider names (e.g. ['openai','gemini','xai']). Omit to use the configured active set. Call list_providers first if you're unsure which names are available.";

const TOOLS: Record<string, { description: string; inputSchema: any; handler: (a: any) => Promise<any> }> = {
  list_providers: {
    description: "List every provider the server knows about and whether each is currently usable (has an API key in .env). Call this first to discover who's on the panel.",
    inputSchema: { type: "object", properties: {} },
    handler: toolListProviders,
  },
  confer: {
    description: "Ask one or more LLMs the same question and return their answers in parallel.",
    inputSchema: { type: "object", properties: {
      question: { type: "string" }, context: { type: "string" },
      providers: { type: "array", items: { type: "string" }, description: PROVIDER_DESC },
    }, required: ["question"] },
    handler: toolConfer,
  },
  debate: {
    description: "Run a bounded multi-round debate across LLMs; moderator synthesises the result.",
    inputSchema: { type: "object", properties: {
      topic: { type: "string" }, context: { type: "string" },
      providers: { type: "array", items: { type: "string" }, description: PROVIDER_DESC + " Needs at least 2." },
      max_rounds: { type: "integer" },
      moderator: { type: "string", description: "Provider name to run the synthesis round. Defaults to config.moderator." },
    }, required: ["topic"] },
    handler: toolDebate,
  },
  plan: {
    description: "Collaborative planning across LLMs with risks + alternatives.",
    inputSchema: { type: "object", properties: {
      goal: { type: "string" }, constraints: { type: "string" }, context: { type: "string" },
      providers: { type: "array", items: { type: "string" }, description: PROVIDER_DESC + " Needs at least 2." },
      moderator: { type: "string" },
    }, required: ["goal"] },
    handler: toolPlan,
  },
  review: {
    description: "Peer-review a code snippet or proposal across one or more LLMs.",
    inputSchema: { type: "object", properties: {
      snippet: { type: "string" }, intent: { type: "string" },
      providers: { type: "array", items: { type: "string" }, description: PROVIDER_DESC },
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
