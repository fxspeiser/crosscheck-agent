# crosscheck-agent

Confer with multiple LLMs from inside Claude Code. `crosscheck-agent` is a
compact, polyglot MCP server that lets Claude ask peers from other model
families (GPT, Grok, Gemini, Mistral, Groq, DeepSeek) to reason, debate,
plan, and peer-review — then hands the synthesised answer back to Claude.

Pick any runtime that matches your stack: **Python**, **TypeScript**,
**Rust**, or **Perl**. All four implementations expose the same tool
surface and read the same config file, so swapping runtimes is zero-friction.

```
       ┌──────────────┐
Claude │ Claude Code  │     MCP      ┌────────────────────────┐
 tool  │  (your IDE)  │ ───────────▶ │  crosscheck-agent MCP  │
 call  │              │   stdio      │  (py / ts / rs / pl)   │
       └──────────────┘              └───────────┬────────────┘
                                                 │ HTTPS
                                                 ▼
                         ┌───────────────────────────────────────┐
                         │ OpenAI · xAI · Gemini · Mistral · Groq │
                         │ DeepSeek · Anthropic (…)              │
                         └───────────────────────────────────────┘
```

## Tools

| Tool      | What it does                                                                 |
|-----------|------------------------------------------------------------------------------|
| `confer`  | Ask N providers the same question in parallel; return every answer.          |
| `debate`  | Bounded round-trip debate; the configured moderator synthesises a result.    |
| `plan`    | Collaborative step-by-step planning with risks + alternatives.               |
| `review`  | Peer code / proposal review across multiple LLMs.                            |

Every run obeys the limits in `crosscheck.config.json`:

- `max_rounds` — hard cap on unsupervised round trips.
- `token_cap` — total token budget spread across providers × rounds.
- `max_time_seconds` — wall-clock deadline enforced per run.

## Quick start

```bash
git clone https://github.com/<you>/crosscheck-agent.git
cd crosscheck-agent

# 1. Interactive setup — writes .env + crosscheck.config.json
bash scripts/setup.sh

# 2. Pick a runtime (only do one):
#    Python — no deps
python3 servers/python/crosscheck_server.py   # sanity-check
claude mcp add crosscheck -- python3 "$PWD/servers/python/crosscheck_server.py"

#    TypeScript
( cd servers/typescript && npm install && npm run build )
claude mcp add crosscheck -- node "$PWD/servers/typescript/dist/index.js"

#    Rust
( cd servers/rust && cargo build --release )
claude mcp add crosscheck -- "$PWD/servers/rust/target/release/crosscheck-agent-rs"

#    Perl
claude mcp add crosscheck -- perl "$PWD/servers/perl/crosscheck_server.pl"
```

Then inside Claude Code:

```
/mcp
# call confer / debate / plan / review
```

## Tuning limits from the terminal

The `scripts/crosscheck` CLI edits `crosscheck.config.json` in place.

```bash
scripts/crosscheck config show
scripts/crosscheck config set max_rounds 5
scripts/crosscheck config set token_cap 16000
scripts/crosscheck config set max_time_seconds 300
scripts/crosscheck config set providers anthropic,openai,xai,gemini
scripts/crosscheck config set moderator openai

scripts/crosscheck providers list
scripts/crosscheck providers enable gemini
scripts/crosscheck providers disable xai

scripts/crosscheck doctor     # audit: keys present, config sane
```

Optionally add this line to your shell rc file to make the CLI globally available:

```bash
export PATH="$PATH:/path/to/crosscheck-agent/scripts"
```

## Providers

| Provider   | Env var              | Default model             | Endpoint                |
|------------|----------------------|---------------------------|-------------------------|
| Anthropic  | `ANTHROPIC_API_KEY`  | `claude-opus-4-5`         | native                  |
| OpenAI     | `OPENAI_API_KEY`     | `gpt-5`                   | Chat Completions        |
| xAI (Grok) | `XAI_API_KEY`        | `grok-4-latest`           | OpenAI-compatible       |
| Google     | `GEMINI_API_KEY`     | `gemini-2.5-pro`          | Gemini API              |
| Mistral    | `MISTRAL_API_KEY`    | `mistral-large-latest`    | OpenAI-compatible       |
| Groq       | `GROQ_API_KEY`       | `llama-3.3-70b-versatile` | OpenAI-compatible       |
| DeepSeek   | `DEEPSEEK_API_KEY`   | `deepseek-chat`           | OpenAI-compatible       |

Any provider without a key in `.env` is silently skipped.

## Configuration

`crosscheck.config.json`:

```json
{
  "max_rounds": 3,
  "token_cap": 8000,
  "max_time_seconds": 120,
  "providers": ["anthropic", "openai", "xai"],
  "moderator": "anthropic",
  "temperature": 0.4,
  "log_transcripts": true,
  "transcript_dir": ".crosscheck/transcripts"
}
```

When `log_transcripts` is on, every conferral / debate is persisted as JSON
under `.crosscheck/transcripts/` (also gitignored).

## Layout

```
crosscheck-agent/
├── .env.example
├── crosscheck.config.example.json
├── scripts/
│   ├── setup.sh            # interactive wizard
│   └── crosscheck          # config + providers CLI
└── servers/
    ├── python/             # stdlib-only MCP server
    ├── typescript/         # Node 20+ MCP server
    ├── rust/               # tokio + reqwest
    └── perl/               # HTTP::Tiny + JSON::PP
```

## Security

- `.env` is gitignored. The setup wizard chmods it to `600`.
- The `crosscheck` CLI never prints API keys, only whether they exist.
- Every runtime reads keys at startup, never writes them anywhere but stderr
  on an HTTP error (which may echo the remote error payload — be mindful if
  you pipe logs to third-party tools).

## Contributing

Issues and PRs welcome. The Python implementation is the reference — any
behavioural change should land there first, then be mirrored across the
other three runtimes. Keep the tool surface (`confer`, `debate`, `plan`,
`review`) stable and dependency-light.

## Credits

Built by [Frank Speiser](https://github.com/fspeiser) with pair-programming
assistance from Claude (Anthropic). Mistakes are Frank's; good ideas are shared.

## License

MIT. See [LICENSE](LICENSE).
