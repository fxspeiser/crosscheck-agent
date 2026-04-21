# crosscheck-agent — Perl runtime

Uses core Perl modules (`HTTP::Tiny`, `JSON::PP`) plus SSL support
(`IO::Socket::SSL` + `Net::SSLeay`) for HTTPS.

## Install SSL deps (macOS)

```bash
cpan IO::Socket::SSL Net::SSLeay
```

Or with Homebrew's perl:

```bash
cpanm IO::Socket::SSL Net::SSLeay
```

## Run it

```bash
perl servers/perl/crosscheck_server.pl
```

## Register with Claude Code

```bash
claude mcp add crosscheck -- perl "$PWD/servers/perl/crosscheck_server.pl"
```
