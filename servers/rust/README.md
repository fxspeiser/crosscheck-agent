# crosscheck-agent — Rust runtime

## Build

```bash
cd servers/rust
cargo build --release
```

Binary ends up at `target/release/crosscheck-agent-rs`.

## Register with Claude Code

```bash
claude mcp add crosscheck -- "$PWD/servers/rust/target/release/crosscheck-agent-rs"
```
