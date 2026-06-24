<!-- togo-header -->
<div align="center">
  <img src=".github/assets/togo-mark.svg" alt="togo" height="64" />
  <h1>togo-framework/ai-rag</h1>
  <p>
    <a href="https://to-go.dev/marketplace"><img src="https://img.shields.io/badge/marketplace-to--go.dev-1FC7DC" alt="marketplace" /></a>
    <a href="https://pkg.go.dev/github.com/togo-framework/ai-rag"><img src="https://pkg.go.dev/badge/github.com/togo-framework/ai-rag.svg" alt="pkg.go.dev" /></a>
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT" />
  </p>
  <p><strong>Part of the <a href="https://to-go.dev">togo</a> framework.</strong></p>
</div>

## Install

```bash
togo install togo-framework/ai-rag
```

<!-- /togo-header -->

# ai-rag

A full **RAG** (retrieval-augmented generation) capability for togo, built on the `ai` plugin.

```bash
togo install togo-framework/ai-rag
```

Ingest documents (chunk → embed via your configured `AI_DRIVER` → vector store), then answer questions with retrieved context.

```go
rag, _ := rag.FromKernel(k)
rag.Ingest(ctx, "doc-1", longText)
answer, sources, _ := rag.Answer(ctx, "What is togo?", 4)
```

Vector store is pluggable via `rag.RegisterStore` (`RAG_STORE`, default `memory` with cosine similarity; add a pgvector store later). Requires the `ai` plugin + a provider (e.g. `ai-openai`).

MIT © ToGO

<!-- togo-sponsors -->
---

<div align="center">
  <h3>Premium sponsors</h3>
  <p>
    <a href="https://id8media.com"><strong>ID8 Media</strong></a> &nbsp;·&nbsp;
    <a href="https://one-studio.co"><strong>One Studio</strong></a>
  </p>
  <p><sub>Support togo — <a href="https://github.com/sponsors/fadymondy">become a sponsor</a>.</sub></p>
</div>
<!-- /togo-sponsors -->
