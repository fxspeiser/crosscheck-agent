#!/usr/bin/env perl
# crosscheck-agent — Perl MCP server.
# Uses only core modules (HTTP::Tiny, JSON::PP). For HTTPS you need
# IO::Socket::SSL + Net::SSLeay installed (both widely packaged).

use strict;
use warnings;
use FindBin qw($Bin);
use File::Spec;
use HTTP::Tiny;
use JSON::PP;
use Time::HiRes qw(time);

my $ROOT = File::Spec->rel2abs(File::Spec->catdir($Bin, '..', '..'));
my $CONFIG      = File::Spec->catfile($ROOT, 'crosscheck.config.json');
my $CONFIG_EX   = File::Spec->catfile($ROOT, 'crosscheck.config.example.json');
my $ENV_FILE    = File::Spec->catfile($ROOT, '.env');
my $TRANS_DIR   = File::Spec->catdir($ROOT, '.crosscheck', 'transcripts');

my %ENV_MAP;

# ------------------------------------------------------------
# .env and config
# ------------------------------------------------------------
sub load_env {
    %ENV_MAP = %ENV;
    return unless -f $ENV_FILE;
    open my $fh, '<', $ENV_FILE or return;
    while (my $line = <$fh>) {
        chomp $line;
        $line =~ s/^\s+|\s+$//g;
        next if $line eq '' || $line =~ /^#/;
        next unless $line =~ /^([^=]+)=(.*)$/;
        my ($k, $v) = ($1, $2);
        $k =~ s/^\s+|\s+$//g; $v =~ s/^\s+|\s+$//g;
        $ENV_MAP{$k} //= $v;
    }
    close $fh;
}

sub load_config {
    my $path = -f $CONFIG ? $CONFIG : $CONFIG_EX;
    open my $fh, '<', $path or die "cannot read $path: $!";
    local $/; my $text = <$fh>; close $fh;
    return decode_json($text);
}

load_env();
my $CFG = load_config();

# ------------------------------------------------------------
# HTTP client
# ------------------------------------------------------------
my $HTTP = HTTP::Tiny->new(timeout => $CFG->{max_time_seconds} // 120);

sub http_post {
    my ($url, $headers, $body) = @_;
    my $resp = $HTTP->request('POST', $url, {
        headers => { 'Content-Type' => 'application/json', %{ $headers || {} } },
        content => encode_json($body),
    });
    if (!$resp->{success}) {
        die "HTTP $resp->{status}: $resp->{content}";
    }
    return decode_json($resp->{content});
}

# ------------------------------------------------------------
# Providers
# ------------------------------------------------------------
sub openai_compatible {
    my ($name, $url, $key_env, $model_env, $default_model) = @_;
    my $key = $ENV_MAP{$key_env} or return undef;
    my $model = $ENV_MAP{$model_env} || $default_model;
    return {
        name  => $name,
        model => $model,
        send  => sub {
            my ($messages, $max_tokens, $temperature) = @_;
            my $resp = http_post($url,
                { Authorization => "Bearer $key" },
                { model => $model, messages => $messages, max_tokens => $max_tokens + 0, temperature => $temperature + 0.0 });
            return ($resp->{choices}[0]{message}{content} // "");
        },
    };
}

sub anthropic_provider {
    my $key = $ENV_MAP{ANTHROPIC_API_KEY} or return undef;
    my $model = $ENV_MAP{ANTHROPIC_MODEL} || 'claude-opus-4-5';
    return {
        name  => 'anthropic',
        model => $model,
        send  => sub {
            my ($messages, $max_tokens, $temperature) = @_;
            my ($system) = map { $_->{content} } grep { $_->{role} eq 'system' } @$messages;
            my @convo = grep { $_->{role} ne 'system' } @$messages;
            my $body = { model => $model, max_tokens => $max_tokens + 0, temperature => $temperature + 0.0, messages => \@convo };
            $body->{system} = $system if defined $system;
            my $resp = http_post('https://api.anthropic.com/v1/messages',
                { 'x-api-key' => $key, 'anthropic-version' => '2023-06-01' },
                $body);
            return join('', map { $_->{text} // '' } @{ $resp->{content} || [] });
        },
    };
}

sub gemini_provider {
    my $key = $ENV_MAP{GEMINI_API_KEY} or return undef;
    my $model = $ENV_MAP{GEMINI_MODEL} || 'gemini-2.5-pro';
    return {
        name  => 'gemini',
        model => $model,
        send  => sub {
            my ($messages, $max_tokens, $temperature) = @_;
            my (@contents, $system);
            for my $m (@$messages) {
                if ($m->{role} eq 'system') { $system = $m->{content}; next; }
                my $role = $m->{role} eq 'user' ? 'user' : 'model';
                push @contents, { role => $role, parts => [{ text => $m->{content} }] };
            }
            my $body = {
                contents => \@contents,
                generationConfig => { maxOutputTokens => $max_tokens + 0, temperature => $temperature + 0.0 },
            };
            $body->{systemInstruction} = { parts => [{ text => $system }] } if defined $system;
            my $url = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$key";
            my $resp = http_post($url, {}, $body);
            my $parts = $resp->{candidates}[0]{content}{parts} || [];
            return join('', map { $_->{text} // '' } @$parts);
        },
    };
}

my %ALL;
for my $p (
    anthropic_provider(),
    openai_compatible('openai',   'https://api.openai.com/v1/chat/completions',      'OPENAI_API_KEY',   'OPENAI_MODEL',   'gpt-5'),
    openai_compatible('xai',      'https://api.x.ai/v1/chat/completions',            'XAI_API_KEY',      'XAI_MODEL',      'grok-4-latest'),
    openai_compatible('mistral',  'https://api.mistral.ai/v1/chat/completions',      'MISTRAL_API_KEY',  'MISTRAL_MODEL',  'mistral-large-latest'),
    openai_compatible('groq',     'https://api.groq.com/openai/v1/chat/completions', 'GROQ_API_KEY',     'GROQ_MODEL',     'llama-3.3-70b-versatile'),
    openai_compatible('deepseek', 'https://api.deepseek.com/v1/chat/completions',    'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL', 'deepseek-chat'),
    gemini_provider(),
) {
    next unless defined $p;
    $ALL{$p->{name}} = $p;
}

sub active_providers {
    my @names = @{ $CFG->{providers} || [] };
    return grep { defined } map { $ALL{$_} } @names;
}

sub per_call_tokens {
    my ($count) = @_;
    my $rounds = ($CFG->{max_rounds} || 3) < 1 ? 1 : $CFG->{max_rounds};
    $count = 1 if $count < 1;
    my $budget = int(($CFG->{token_cap} || 8000) / ($rounds * $count));
    return $budget < 256 ? 256 : $budget;
}

sub write_transcript {
    my ($kind, $payload) = @_;
    return undef unless $CFG->{log_transcripts} // 1;
    unless (-d $TRANS_DIR) {
        require File::Path;
        File::Path::make_path($TRANS_DIR);
    }
    my @t = gmtime(time);
    my $stamp = sprintf("%04d%02d%02dT%02d%02d%02d", $t[5]+1900, $t[4]+1, $t[3], $t[2], $t[1], $t[0]);
    my $path = File::Spec->catfile($TRANS_DIR, "$stamp-$kind.json");
    open my $fh, '>', $path or return undef;
    print $fh JSON::PP->new->pretty->encode($payload);
    close $fh;
    my $rel = File::Spec->abs2rel($path, $ROOT);
    return $rel;
}

sub ask_one {
    my ($p, $messages, $deadline, $per_call) = @_;
    if (time >= $deadline) {
        return { provider => $p->{name}, model => $p->{model}, error => "time budget exhausted" };
    }
    my $temp = $CFG->{temperature} // 0.4;
    my $out = eval { $p->{send}->($messages, $per_call, $temp) };
    if ($@) {
        my $err = $@; chomp $err;
        return { provider => $p->{name}, model => $p->{model}, error => "$err" };
    }
    return { provider => $p->{name}, model => $p->{model}, response => $out };
}

# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
sub tool_confer {
    my ($args) = @_;
    my $question = $args->{question} // '';
    my $context  = $args->{context}  // '';
    my @names = $args->{providers} ? @{ $args->{providers} } : map { $_->{name} } active_providers();
    my @picked = grep { defined } map { $ALL{$_} } @names;
    return { error => "no active providers have API keys in .env" } unless @picked;

    my @messages = (
        { role => 'system', content => "You are part of a panel of LLMs consulted by an engineer inside Claude Code. Answer directly, cite assumptions, stay crisp." },
    );
    push @messages, { role => 'user', content => "CONTEXT:\n$context" } if $context;
    push @messages, { role => 'user', content => $question };

    my $deadline = time + ($CFG->{max_time_seconds} // 120);
    my $per_call = per_call_tokens(scalar @picked);
    my @answers = map { ask_one($_, \@messages, $deadline, $per_call) } @picked;

    my $result = { tool => 'confer', question => $question, answers => \@answers };
    my $path = write_transcript('confer', $result);
    $result->{transcript} = $path if defined $path;
    return $result;
}

sub tool_debate {
    my ($args) = @_;
    my $topic   = $args->{topic}   // '';
    my $context = $args->{context} // '';
    my @names = $args->{providers} ? @{ $args->{providers} } : map { $_->{name} } active_providers();
    my @picked = grep { defined } map { $ALL{$_} } @names;
    return { error => "debate needs at least 2 active providers" } unless @picked >= 2;

    my $max_rounds = $args->{max_rounds} // $CFG->{max_rounds} // 3;
    my $deadline = time + ($CFG->{max_time_seconds} // 120);
    my $per_call = per_call_tokens(scalar @picked);
    my @transcript;

    for my $rnd (1 .. $max_rounds) {
        last if time >= $deadline - 1;
        my @round_messages = (
            { role => 'system', content => "You are debating peers from other model families. Round $rnd/$max_rounds. Disagree where warranted, concede where right, keep replies short." },
        );
        push @round_messages, { role => 'user', content => "CONTEXT:\n$context" } if $context;
        push @round_messages, { role => 'user', content => "TOPIC: $topic" };
        if (@transcript) {
            my $prior = join("\n\n", map {
                "[$_->{provider} — round $_->{round}]\n" . ($_->{response} // '(error)')
            } @transcript);
            push @round_messages, { role => 'user', content => "PRIOR TURNS:\n$prior" };
        }
        for my $p (@picked) {
            last if time >= $deadline - 1;
            my $entry = ask_one($p, \@round_messages, $deadline, $per_call);
            $entry->{round} = $rnd;
            push @transcript, $entry;
        }
    }

    my $mod_name = $args->{moderator} // $CFG->{moderator} // 'anthropic';
    my $moderator = $ALL{$mod_name} // $picked[0];
    my $synthesis = undef;
    if ($moderator && time < $deadline - 1) {
        my $condensed = join("\n\n", map {
            "[$_->{provider} — round $_->{round}]\n" . ($_->{response} // '(error)')
        } @transcript);
        $synthesis = ask_one($moderator, [
            { role => 'system', content => "You are the moderator. Synthesise the debate into a single grounded recommendation." },
            { role => 'user',   content => "TOPIC: $topic\n\nTRANSCRIPT:\n$condensed" },
        ], $deadline, $per_call);
    }

    my $rounds_completed = 0;
    for my $e (@transcript) { $rounds_completed = $e->{round} if $e->{round} > $rounds_completed; }

    my $result = {
        tool => 'debate',
        topic => $topic,
        rounds_completed => $rounds_completed,
        transcript => \@transcript,
        synthesis  => $synthesis,
    };
    my $path = write_transcript('debate', $result);
    $result->{transcript_path} = $path if defined $path;
    return $result;
}

sub tool_plan {
    my ($args) = @_;
    my $goal = $args->{goal} // '';
    my $constraints = $args->{constraints} // '';
    my $topic = "We need a step-by-step plan to achieve this goal.\n\nGOAL: $goal\n\nCONSTRAINTS: "
              . ($constraints || '(none stated)')
              . "\n\nReturn: (1) the plan as numbered steps, (2) risks, (3) alternatives considered.";
    return tool_debate({ topic => $topic, context => ($args->{context} // '') });
}

sub tool_review {
    my ($args) = @_;
    my $snippet = $args->{snippet} // '';
    my $intent  = $args->{intent}  || '(not stated)';
    my $question = "Review the following code/proposal as peers. Call out bugs, smells, missed edge cases, and suggest concrete changes.\n\n"
                 . "INTENT: $intent\n\nSNIPPET:\n```\n$snippet\n```";
    return tool_confer({ question => $question });
}

# ------------------------------------------------------------
# MCP JSON-RPC over stdio
# ------------------------------------------------------------
my %TOOLS = (
    confer => {
        description => "Ask multiple LLMs the same question and return all answers in parallel.",
        inputSchema => {
            type => 'object',
            properties => {
                question  => { type => 'string' },
                context   => { type => 'string' },
                providers => { type => 'array', items => { type => 'string' } },
            },
            required => [ 'question' ],
        },
        handler => \&tool_confer,
    },
    debate => {
        description => "Run a bounded multi-round debate across LLMs; moderator synthesises the result.",
        inputSchema => {
            type => 'object',
            properties => {
                topic      => { type => 'string' },
                context    => { type => 'string' },
                providers  => { type => 'array', items => { type => 'string' } },
                max_rounds => { type => 'integer' },
                moderator  => { type => 'string' },
            },
            required => [ 'topic' ],
        },
        handler => \&tool_debate,
    },
    plan => {
        description => "Collaborative planning across LLMs with risks + alternatives.",
        inputSchema => {
            type => 'object',
            properties => {
                goal        => { type => 'string' },
                constraints => { type => 'string' },
                context     => { type => 'string' },
            },
            required => [ 'goal' ],
        },
        handler => \&tool_plan,
    },
    review => {
        description => "Peer-review a code snippet or proposal across LLMs.",
        inputSchema => {
            type => 'object',
            properties => {
                snippet => { type => 'string' },
                intent  => { type => 'string' },
            },
            required => [ 'snippet' ],
        },
        handler => \&tool_review,
    },
);

sub rpc_result { my ($id, $r) = @_; return { jsonrpc => '2.0', id => $id, result => $r }; }
sub rpc_error  { my ($id, $c, $m) = @_; return { jsonrpc => '2.0', id => $id, error => { code => $c+0, message => $m } }; }

sub handle {
    my ($req) = @_;
    my $method = $req->{method} // '';
    my $id     = $req->{id};
    my $params = $req->{params} || {};
    if ($method eq 'initialize') {
        return rpc_result($id, {
            protocolVersion => '2024-11-05',
            capabilities    => { tools => {} },
            serverInfo      => { name => 'crosscheck-agent', version => '0.1.0' },
        });
    }
    return undef if $method eq 'notifications/initialized';
    if ($method eq 'tools/list') {
        my @list;
        for my $name (keys %TOOLS) {
            push @list, {
                name        => $name,
                description => $TOOLS{$name}{description},
                inputSchema => $TOOLS{$name}{inputSchema},
            };
        }
        return rpc_result($id, { tools => \@list });
    }
    if ($method eq 'tools/call') {
        my $name = $params->{name} // '';
        my $tool = $TOOLS{$name};
        return rpc_error($id, -32601, "unknown tool: $name") unless $tool;
        my $out = eval { $tool->{handler}->($params->{arguments} || {}) };
        return rpc_error($id, -32000, "$@") if $@;
        return rpc_result($id, {
            content => [ { type => 'text', text => JSON::PP->new->pretty->encode($out) } ],
        });
    }
    return undef unless defined $id;
    return rpc_error($id, -32601, "unknown method: $method");
}

# stdio loop
$| = 1;
my $json = JSON::PP->new;
while (my $line = <STDIN>) {
    chomp $line;
    $line =~ s/^\s+|\s+$//g;
    next if $line eq '';
    my $req = eval { $json->decode($line) } or next;
    my $resp = handle($req);
    if (defined $resp) {
        print $json->encode($resp), "\n";
    }
}
