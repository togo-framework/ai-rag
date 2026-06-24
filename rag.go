// Package rag is the togo RAG capability — built on the `ai` plugin. It ingests
// documents (chunk → embed via the configured AI provider → vector store) and
// answers questions by retrieving the most relevant chunks and generating with
// the LLM. The vector store is pluggable (RegisterStore); the default is in-memory.
package rag

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/togo-framework/ai"
	"github.com/togo-framework/togo"
)

// Chunk is a stored, embedded slice of a document.
type Chunk struct {
	ID     string    `json:"id"`
	DocID  string    `json:"doc_id"`
	Text   string    `json:"text"`
	Score  float32   `json:"score,omitempty"`
	Vector []float32 `json:"-"`
}

// Store is a pluggable vector store.
type Store interface {
	Upsert(ctx context.Context, chunks []Chunk) error
	Search(ctx context.Context, vector []float32, topK int) ([]Chunk, error)
	Delete(ctx context.Context, docID string) error
}

// StoreFactory builds a Store from the kernel/env.
type StoreFactory func(k *togo.Kernel) (Store, error)

var (
	storeMu sync.RWMutex
	stores  = map[string]StoreFactory{"memory": func(*togo.Kernel) (Store, error) { return newMemStore(), nil }}
)

// RegisterStore registers a vector-store driver (e.g. a pgvector plugin's init()).
func RegisterStore(name string, f StoreFactory) {
	storeMu.Lock()
	stores[name] = f
	storeMu.Unlock()
}

// Service is the kernel-bound RAG service.
type Service struct {
	ai        *ai.Service
	store     Store
	chunkSize int
	overlap   int
}

func init() {
	togo.RegisterProviderFunc("ai-rag", togo.PriorityService, func(k *togo.Kernel) error {
		svc, ok := ai.FromKernel(k)
		if !ok {
			return errors.New("ai-rag: the `ai` plugin is not active (install togo-framework/ai + a provider)")
		}
		name := envOr(k, "RAG_STORE", "memory")
		storeMu.RLock()
		f, ok := stores[name]
		storeMu.RUnlock()
		if !ok {
			return fmt.Errorf("ai-rag: unknown store %q", name)
		}
		st, err := f(k)
		if err != nil {
			return err
		}
		k.Set("ai-rag", &Service{ai: svc, store: st, chunkSize: 1000, overlap: 150})
		return nil
	})
}

func envOr(k *togo.Kernel, key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// FromKernel returns the RAG service bound to the kernel.
func FromKernel(k *togo.Kernel) (*Service, bool) {
	v, ok := k.Get("ai-rag")
	if !ok {
		return nil, false
	}
	s, ok := v.(*Service)
	return s, ok
}

// Ingest chunks `text`, embeds each chunk, and upserts it under docID.
func (s *Service) Ingest(ctx context.Context, docID, text string) (int, error) {
	parts := chunkText(text, s.chunkSize, s.overlap)
	if len(parts) == 0 {
		return 0, nil
	}
	emb, err := s.ai.Embed(ctx, ai.EmbedRequest{Inputs: parts})
	if err != nil {
		return 0, err
	}
	if len(emb.Vectors) != len(parts) {
		return 0, fmt.Errorf("ai-rag: embedding count %d != chunk count %d", len(emb.Vectors), len(parts))
	}
	chunks := make([]Chunk, len(parts))
	for i, p := range parts {
		chunks[i] = Chunk{ID: fmt.Sprintf("%s#%d", docID, i), DocID: docID, Text: p, Vector: emb.Vectors[i]}
	}
	return len(chunks), s.store.Upsert(ctx, chunks)
}

// Retrieve returns the topK chunks most relevant to the query.
func (s *Service) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	if topK <= 0 {
		topK = 4
	}
	emb, err := s.ai.Embed(ctx, ai.EmbedRequest{Inputs: []string{query}})
	if err != nil {
		return nil, err
	}
	if len(emb.Vectors) == 0 {
		return nil, errors.New("ai-rag: no embedding for query")
	}
	return s.store.Search(ctx, emb.Vectors[0], topK)
}

// Answer retrieves context and generates an answer with the LLM.
func (s *Service) Answer(ctx context.Context, query string, topK int) (string, []Chunk, error) {
	chunks, err := s.Retrieve(ctx, query, topK)
	if err != nil {
		return "", nil, err
	}
	var ctxBuf strings.Builder
	for i, c := range chunks {
		fmt.Fprintf(&ctxBuf, "[%d] %s\n\n", i+1, c.Text)
	}
	msgs := []ai.Message{
		{Role: ai.RoleSystem, Content: "Answer the question using ONLY the provided context. Cite sources as [n]. If the answer isn't in the context, say you don't know."},
		{Role: ai.RoleUser, Content: "Context:\n" + ctxBuf.String() + "\nQuestion: " + query},
	}
	resp, err := s.ai.Chat(ctx, ai.ChatRequest{Messages: msgs})
	if err != nil {
		return "", chunks, err
	}
	return resp.Content, chunks, nil
}

// Delete removes a document's chunks.
func (s *Service) Delete(ctx context.Context, docID string) error { return s.store.Delete(ctx, docID) }

// chunkText splits text into ~size-rune windows with `overlap` rune overlap.
func chunkText(text string, size, overlap int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	r := []rune(text)
	if len(r) <= size {
		return []string{text}
	}
	var out []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for i := 0; i < len(r); i += step {
		end := i + size
		if end > len(r) {
			end = len(r)
		}
		out = append(out, strings.TrimSpace(string(r[i:end])))
		if end == len(r) {
			break
		}
	}
	return out
}

// ── in-memory store (cosine similarity) ──
type memStore struct {
	mu sync.RWMutex
	m  map[string]Chunk
}

func newMemStore() *memStore { return &memStore{m: map[string]Chunk{}} }

func (s *memStore) Upsert(_ context.Context, chunks []Chunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range chunks {
		s.m[c.ID] = c
	}
	return nil
}

func (s *memStore) Search(_ context.Context, q []float32, topK int) ([]Chunk, error) {
	s.mu.RLock()
	cand := make([]Chunk, 0, len(s.m))
	for _, c := range s.m {
		c.Score = cosine(q, c.Vector)
		cand = append(cand, c)
	}
	s.mu.RUnlock()
	sort.Slice(cand, func(i, j int) bool { return cand[i].Score > cand[j].Score })
	if len(cand) > topK {
		cand = cand[:topK]
	}
	return cand, nil
}

func (s *memStore) Delete(_ context.Context, docID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for id, c := range s.m {
		if c.DocID == docID {
			delete(s.m, id)
		}
	}
	return nil
}

func cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
