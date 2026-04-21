# crosscheck-agent — Python runtime

Zero external dependencies — uses only the Python stdlib.

## Run it

```bash
python3 servers/python/crosscheck_server.py
```

## Register with Claude Code

```bash
claude mcp add crosscheck -- python3 "$PWD/servers/python/crosscheck_server.py"
```

Then from any Claude Code session:

```
/mcp
```

and call `confer`, `debate`, `plan`, or `review`.
