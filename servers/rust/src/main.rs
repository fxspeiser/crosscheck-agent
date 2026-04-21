// crosscheck-agent — Rust MCP server.
// Mirrors the Python reference implementation. Reads crosscheck.config.json
// and .env from the repo root (two levels up from this crate).

use std::{
    collections::HashMap,
    env,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader},
    time::Instant,
};

// ------------------------------------------------------------
// Config & .env
// ------------------------------------------------------------
#[derive(Debug, Deserialize, Clone)]
struct Config {
    max_rounds: u32,
    token_cap: u32,
    max_time_seconds: u64,
    providers: Vec<String>,
    moderator: String,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_true")]
    log_transcripts: bool,
}
fn default_temp() -> f32 { 0.4 }
fn default_true() -> bool { true }

fn root_dir() -> PathBuf {
    // Find the repo root: walk up from CARGO_MANIFEST_DIR until we hit
    // crosscheck.config.example.json. Falls back to two levels up.
    let start = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap());
    let mut p: &Path = &start;
    loop {
        if p.join("crosscheck.config.example.json").exists() {
            return p.to_path_buf();
        }
        match p.parent() {
            Some(parent) => p = parent,
            None => return start.parent().and_then(Path::parent).map(Path::to_path_buf).unwrap_or(start.clone()),
        }
    }
}

fn load_env(root: &Path) -> HashMap<String, String> {
    let mut env_map: HashMap<String, String> = env::vars().collect();
    let path = root.join(".env");
    if let Ok(text) = fs::read_to_string(&path) {
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some(idx) = line.find('=') {
                let k = line[..idx].trim().to_string();
                let v = line[idx+1..].trim().to_string();
                env_map.entry(k).or_insert(v);
            }
        }
    }
    env_map
}

fn load_config(root: &Path) -> Config {
    let p1 = root.join("crosscheck.config.json");
    let p2 = root.join("crosscheck.config.example.json");
    let path = if p1.exists() { p1 } else { p2 };
    let text = fs::read_to_string(&path).expect("read config");
    serde_json::from_str(&text).expect("parse config")
}

// ------------------------------------------------------------
// Providers
// ------------------------------------------------------------
#[derive(Debug, Clone, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Clone)]
enum ProviderKind {
    OpenAICompatible { url: String, bearer: String },
    Anthropic { key: String },
    Gemini { key: String },
}

#[derive(Clone)]
struct Provider {
    name: String,
    model: String,
    kind: ProviderKind,
}

fn build_providers(env: &HashMap<String, String>) -> HashMap<String, Provider> {
    let mut out = HashMap::new();

    let mut add_openai = |name: &str, url: &str, key_env: &str, model_env: &str, default_model: &str| {
        if let Some(key) = env.get(key_env) {
            let model = env.get(model_env).cloned().unwrap_or_else(|| default_model.to_string());
            out.insert(name.to_string(), Provider {
                name: name.to_string(), model,
                kind: ProviderKind::OpenAICompatible { url: url.to_string(), bearer: key.clone() },
            });
        }
    };

    add_openai("openai",   "https://api.openai.com/v1/chat/completions",          "OPENAI_API_KEY",   "OPENAI_MODEL",   "gpt-5");
    add_openai("xai",      "https://api.x.ai/v1/chat/completions",                "XAI_API_KEY",      "XAI_MODEL",      "grok-4-latest");
    add_openai("mistral",  "https://api.mistral.ai/v1/chat/completions",          "MISTRAL_API_KEY",  "MISTRAL_MODEL",  "mistral-large-latest");
    add_openai("groq",     "https://api.groq.com/openai/v1/chat/completions",     "GROQ_API_KEY",     "GROQ_MODEL",     "llama-3.3-70b-versatile");
    add_openai("deepseek", "https://api.deepseek.com/v1/chat/completions",        "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat");

    if let Some(key) = env.get("ANTHROPIC_API_KEY") {
        let model = env.get("ANTHROPIC_MODEL").cloned().unwrap_or_else(|| "claude-opus-4-5".into());
        out.insert("anthropic".into(), Provider {
            name: "anthropic".into(), model,
            kind: ProviderKind::Anthropic { key: key.clone() },
        });
    }
    if let Some(key) = env.get("GEMINI_API_KEY") {
        let model = env.get("GEMINI_MODEL").cloned().unwrap_or_else(|| "gemini-2.5-pro".into());
        out.insert("gemini".into(), Provider {
            name: "gemini".into(), model,
            kind: ProviderKind::Gemini { key: key.clone() },
        });
    }
    out
}

async fn send(client: &Client, p: &Provider, messages: &[Message], max_tokens: u32, temperature: f32, timeout: Duration) -> Result<String, String> {
    match &p.kind {
        ProviderKind::OpenAICompatible { url, bearer } => {
            let body = json!({
                "model": p.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            });
            let resp = client.post(url).timeout(timeout)
                .bearer_auth(bearer)
                .json(&body)
                .send().await.map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                let code = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("HTTP {}: {}", code, text));
            }
            let v: Value = resp.json().await.map_err(|e| e.to_string())?;
            Ok(v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string())
        }
        ProviderKind::Anthropic { key } => {
            let system = messages.iter().find(|m| m.role == "system").map(|m| m.content.clone());
            let convo: Vec<_> = messages.iter().filter(|m| m.role != "system").collect();
            let mut body = json!({
                "model": p.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": convo,
            });
            if let Some(s) = system { body["system"] = json!(s); }
            let resp = client.post("https://api.anthropic.com/v1/messages").timeout(timeout)
                .header("x-api-key", key)
                .header("anthropic-version", "2023-06-01")
                .json(&body)
                .send().await.map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                let code = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("HTTP {}: {}", code, text));
            }
            let v: Value = resp.json().await.map_err(|e| e.to_string())?;
            let parts = v["content"].as_array().cloned().unwrap_or_default();
            Ok(parts.iter().filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join(""))
        }
        ProviderKind::Gemini { key } => {
            let mut contents: Vec<Value> = Vec::new();
            let mut system: Option<String> = None;
            for m in messages {
                if m.role == "system" { system = Some(m.content.clone()); continue; }
                let role = if m.role == "user" { "user" } else { "model" };
                contents.push(json!({"role": role, "parts": [{"text": m.content}]}));
            }
            let mut body = json!({
                "contents": contents,
                "generationConfig": { "maxOutputTokens": max_tokens, "temperature": temperature },
            });
            if let Some(s) = system { body["systemInstruction"] = json!({"parts":[{"text": s}]}); }
            let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", p.model, key);
            let resp = client.post(url).timeout(timeout)
                .json(&body)
                .send().await.map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                let code = resp.status().as_u16();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("HTTP {}: {}", code, text));
            }
            let v: Value = resp.json().await.map_err(|e| e.to_string())?;
            let parts = v["candidates"][0]["content"]["parts"].as_array().cloned().unwrap_or_default();
            Ok(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join(""))
        }
    }
}

// ------------------------------------------------------------
// Tools
// ------------------------------------------------------------
struct Ctx {
    cfg: Config,
    providers: HashMap<String, Provider>,
    root: PathBuf,
    client: Client,
}

fn per_call_tokens(cfg: &Config, active_count: usize) -> u32 {
    let rounds = cfg.max_rounds.max(1);
    let count = active_count.max(1) as u32;
    (cfg.token_cap / (rounds * count)).max(256)
}

fn active_providers(ctx: &Ctx) -> Vec<Provider> {
    ctx.cfg.providers.iter().filter_map(|n| ctx.providers.get(n).cloned()).collect()
}

fn write_transcript(ctx: &Ctx, kind: &str, payload: &Value) -> Option<String> {
    if !ctx.cfg.log_transcripts { return None; }
    let dir = ctx.root.join(".crosscheck").join("transcripts");
    let _ = fs::create_dir_all(&dir);
    let stamp = chrono_stamp();
    let path = dir.join(format!("{}-{}.json", stamp, kind));
    fs::write(&path, serde_json::to_string_pretty(payload).unwrap()).ok()?;
    path.strip_prefix(&ctx.root).ok().map(|p| p.to_string_lossy().to_string())
}

fn chrono_stamp() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    format!("{}", now)
}

async fn ask_one(ctx: &Ctx, p: &Provider, messages: &[Message], deadline: Instant, per_call: u32) -> Value {
    let now = Instant::now();
    if now >= deadline {
        return json!({"provider": p.name, "model": p.model, "error": "time budget exhausted"});
    }
    let remaining = deadline.saturating_duration_since(now);
    match send(&ctx.client, p, messages, per_call, ctx.cfg.temperature, remaining).await {
        Ok(text) => json!({"provider": p.name, "model": p.model, "response": text}),
        Err(e)   => json!({"provider": p.name, "model": p.model, "error": e}),
    }
}

async fn tool_confer(ctx: Arc<Ctx>, args: Value) -> Value {
    let question = args["question"].as_str().unwrap_or("").to_string();
    let context = args["context"].as_str().unwrap_or("").to_string();
    let names: Vec<String> = args["providers"].as_array().map(|arr|
        arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
    ).unwrap_or_else(|| active_providers(&ctx).iter().map(|p| p.name.clone()).collect());
    let picked: Vec<Provider> = names.iter().filter_map(|n| ctx.providers.get(n).cloned()).collect();
    if picked.is_empty() {
        return json!({"error": "no active providers have API keys in .env"});
    }

    let mut messages = vec![Message {
        role: "system".into(),
        content: "You are part of a panel of LLMs consulted by an engineer inside Claude Code. Answer directly, cite assumptions, stay crisp.".into(),
    }];
    if !context.is_empty() {
        messages.push(Message { role: "user".into(), content: format!("CONTEXT:\n{}", context) });
    }
    messages.push(Message { role: "user".into(), content: question.clone() });

    let deadline = Instant::now() + Duration::from_secs(ctx.cfg.max_time_seconds);
    let per_call = per_call_tokens(&ctx.cfg, picked.len());
    let futs = picked.iter().map(|p| ask_one(&ctx, p, &messages, deadline, per_call));
    let answers: Vec<Value> = join_all(futs).await;

    let mut out = json!({"tool": "confer", "question": question, "answers": answers});
    if let Some(path) = write_transcript(&ctx, "confer", &out) {
        out["transcript"] = json!(path);
    }
    out
}

async fn tool_debate(ctx: Arc<Ctx>, args: Value) -> Value {
    let topic = args["topic"].as_str().unwrap_or("").to_string();
    let context = args["context"].as_str().unwrap_or("").to_string();
    let names: Vec<String> = args["providers"].as_array().map(|arr|
        arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
    ).unwrap_or_else(|| active_providers(&ctx).iter().map(|p| p.name.clone()).collect());
    let picked: Vec<Provider> = names.iter().filter_map(|n| ctx.providers.get(n).cloned()).collect();
    if picked.len() < 2 {
        return json!({"error": "debate needs at least 2 active providers"});
    }
    let max_rounds = args["max_rounds"].as_u64().unwrap_or(ctx.cfg.max_rounds as u64) as u32;
    let deadline = Instant::now() + Duration::from_secs(ctx.cfg.max_time_seconds);
    let per_call = per_call_tokens(&ctx.cfg, picked.len());
    let mut transcript: Vec<Value> = Vec::new();

    for rnd in 1..=max_rounds {
        if Instant::now() >= deadline { break; }
        let mut round_messages = vec![
            Message { role: "system".into(), content: format!("You are debating peers from other model families. Round {}/{}. Disagree where warranted, concede where right, keep replies short.", rnd, max_rounds) },
        ];
        if !context.is_empty() {
            round_messages.push(Message { role: "user".into(), content: format!("CONTEXT:\n{}", context) });
        }
        round_messages.push(Message { role: "user".into(), content: format!("TOPIC: {}", topic) });
        if !transcript.is_empty() {
            let prior = transcript.iter().map(|e| format!(
                "[{} — round {}]\n{}",
                e["provider"].as_str().unwrap_or("?"),
                e["round"].as_u64().unwrap_or(0),
                e["response"].as_str().unwrap_or("(error)"),
            )).collect::<Vec<_>>().join("\n\n");
            round_messages.push(Message { role: "user".into(), content: format!("PRIOR TURNS:\n{}", prior) });
        }
        for p in &picked {
            if Instant::now() >= deadline { break; }
            let mut entry = ask_one(&ctx, p, &round_messages, deadline, per_call).await;
            entry["round"] = json!(rnd);
            transcript.push(entry);
        }
    }

    let mod_name = args["moderator"].as_str().unwrap_or(&ctx.cfg.moderator);
    let moderator = ctx.providers.get(mod_name).cloned().or_else(|| picked.first().cloned());
    let mut synthesis = Value::Null;
    if let Some(mod_p) = moderator {
        if Instant::now() < deadline {
            let condensed = transcript.iter().map(|e| format!(
                "[{} — round {}]\n{}",
                e["provider"].as_str().unwrap_or("?"),
                e["round"].as_u64().unwrap_or(0),
                e["response"].as_str().unwrap_or("(error)"),
            )).collect::<Vec<_>>().join("\n\n");
            let synth_msgs = vec![
                Message { role: "system".into(), content: "You are the moderator. Synthesise the debate into a single grounded recommendation.".into() },
                Message { role: "user".into(), content: format!("TOPIC: {}\n\nTRANSCRIPT:\n{}", topic, condensed) },
            ];
            synthesis = ask_one(&ctx, &mod_p, &synth_msgs, deadline, per_call).await;
        }
    }

    let rounds_completed = transcript.iter().filter_map(|e| e["round"].as_u64()).max().unwrap_or(0);
    let mut out = json!({
        "tool": "debate",
        "topic": topic,
        "rounds_completed": rounds_completed,
        "transcript": transcript,
        "synthesis": synthesis,
    });
    if let Some(path) = write_transcript(&ctx, "debate", &out) {
        out["transcript_path"] = json!(path);
    }
    out
}

async fn tool_plan(ctx: Arc<Ctx>, args: Value) -> Value {
    let goal = args["goal"].as_str().unwrap_or("").to_string();
    let constraints = args["constraints"].as_str().unwrap_or("").to_string();
    let topic = format!(
        "We need a step-by-step plan to achieve this goal.\n\nGOAL: {}\n\nCONSTRAINTS: {}\n\nReturn: (1) the plan as numbered steps, (2) risks, (3) alternatives considered.",
        goal, if constraints.is_empty() { "(none stated)".into() } else { constraints },
    );
    tool_debate(ctx, json!({"topic": topic, "context": args["context"].as_str().unwrap_or("")})).await
}

async fn tool_review(ctx: Arc<Ctx>, args: Value) -> Value {
    let snippet = args["snippet"].as_str().unwrap_or("");
    let intent = args["intent"].as_str().unwrap_or("(not stated)");
    let question = format!(
        "Review the following code/proposal as peers. Call out bugs, smells, missed edge cases, and suggest concrete changes.\n\nINTENT: {}\n\nSNIPPET:\n```\n{}\n```",
        intent, snippet,
    );
    tool_confer(ctx, json!({"question": question})).await
}

// ------------------------------------------------------------
// MCP JSON-RPC loop
// ------------------------------------------------------------
fn tool_schemas() -> Value {
    json!([
        {
            "name": "confer",
            "description": "Ask multiple LLMs the same question and return all answers in parallel.",
            "inputSchema": { "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                    "providers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["question"] }
        },
        {
            "name": "debate",
            "description": "Run a bounded multi-round debate across LLMs; moderator synthesises the result.",
            "inputSchema": { "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "context": {"type": "string"},
                    "providers": {"type": "array", "items": {"type": "string"}},
                    "max_rounds": {"type": "integer"},
                    "moderator": {"type": "string"}
                },
                "required": ["topic"] }
        },
        {
            "name": "plan",
            "description": "Collaborative planning across LLMs with risks + alternatives.",
            "inputSchema": { "type": "object",
                "properties": { "goal": {"type": "string"}, "constraints": {"type": "string"}, "context": {"type": "string"} },
                "required": ["goal"] }
        },
        {
            "name": "review",
            "description": "Peer-review a code snippet or proposal across LLMs.",
            "inputSchema": { "type": "object",
                "properties": { "snippet": {"type": "string"}, "intent": {"type": "string"} },
                "required": ["snippet"] }
        }
    ])
}

async fn handle(ctx: Arc<Ctx>, req: Value) -> Option<Value> {
    let method = req["method"].as_str().unwrap_or("");
    let id = req.get("id").cloned();
    match method {
        "initialize" => Some(json!({
            "jsonrpc": "2.0", "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "crosscheck-agent", "version": "0.1.0" },
            }
        })),
        "notifications/initialized" => None,
        "tools/list" => Some(json!({
            "jsonrpc": "2.0", "id": id,
            "result": { "tools": tool_schemas() }
        })),
        "tools/call" => {
            let name = req["params"]["name"].as_str().unwrap_or("").to_string();
            let args = req["params"]["arguments"].clone();
            let out = match name.as_str() {
                "confer" => tool_confer(ctx, args).await,
                "debate" => tool_debate(ctx, args).await,
                "plan"   => tool_plan(ctx, args).await,
                "review" => tool_review(ctx, args).await,
                _ => return Some(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("unknown tool: {}", name)}})),
            };
            Some(json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "content": [{"type":"text", "text": serde_json::to_string_pretty(&out).unwrap_or_default()}] }
            }))
        }
        _ => {
            if id.is_none() { None } else {
                Some(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":format!("unknown method: {}", method)}}))
            }
        }
    }
}

#[tokio::main]
async fn main() {
    let root = root_dir();
    let env_map = load_env(&root);
    let cfg = load_config(&root);
    let providers = build_providers(&env_map);
    let client = Client::builder().build().expect("reqwest client");
    let ctx = Arc::new(Ctx { cfg, providers, root, client });

    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = io::stdout();

    while let Ok(Some(line)) = reader.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(resp) = handle(ctx.clone(), req).await {
            let _ = stdout.write_all(serde_json::to_string(&resp).unwrap().as_bytes()).await;
            let _ = stdout.write_all(b"\n").await;
            let _ = stdout.flush().await;
        }
    }
}
