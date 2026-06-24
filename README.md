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
