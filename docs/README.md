# ai-rag — documentation

RAG capability for togo (on the ai plugin)

## Overview

Package rag is the togo RAG capability — built on the `ai` plugin. It ingests
documents (chunk → embed via the configured AI provider → vector store) and
answers questions by retrieving the most relevant chunks and generating with
the LLM. The vector store is pluggable (RegisterStore); the default is in-memory.

## Install

```bash
togo install togo-framework/ai-rag
```

A capability plugin — it self-registers on boot; no driver selector needed.

## Configuration

Environment variables read by this plugin (extracted from the source — see the gateway/provider docs for each value):

_No environment variables read directly (uses the kernel/base config or the app DB)._

## Usage

```go
r := rag.FromKernel(k)
r.Ingest(ctx, docs)            // chunk -> embed -> store (pluggable vector store)
ans, _ := r.Generate(ctx, query) // retrieve top-k + answer with [n] citations
```

## Links

- Marketplace: https://to-go.dev/marketplace
- Source: https://github.com/togo-framework/ai-rag
- Full README: ../README.md
