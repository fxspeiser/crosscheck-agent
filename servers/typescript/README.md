# crosscheck-agent — TypeScript runtime

Uses Node 20+ built-in `fetch`. Zero runtime dependencies; TypeScript is a devDep.

## Run it

```bash
cd servers/typescript
npm install
npm run build
node dist/index.js
```

Or without a build step (uses `tsx`):

```bash
npm run dev
```

## Register with Claude Code

```bash
claude mcp add crosscheck -- node "$PWD/servers/typescript/dist/index.js"
```
