package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"slices"
	"strings"
	"sync/atomic"
	"time"
)

// LlamaCppConfig configures the llama.cpp embedder, which calls a local
// llama-server's /embedding endpoint for true semantic vectors.
type LlamaCppConfig struct {
	ServerURL  string // llama-server base URL, e.g. http://127.0.0.1:8080
	Model      string // optional model id (when the server multiplexes)
	Dim        int    // a hint: the dimension the server actually returns wins
	APIKey     string // optional; sent as "Authorization: Bearer <key>"
	Timeout    time.Duration
	MaxRetries int
	BatchSize  int
	Logger     *slog.Logger // optional; nil discards
}

// LlamaCppEmbedder is the DEFAULT embedder: it calls a local
// llama.cpp server over HTTP so recall is genuinely semantic out of the box
// (no cloud hop — the server runs on the host). Point ServerURL at your
// llama-server; see the README for the exact flags.
//
// Transient failures (network errors, 429 and 5xx other than 501) are
// retried. After a connection-level failure or a 501 the server is skipped
// for a short cool-down and calls fail fast, so callers can fall back
// immediately.
type LlamaCppEmbedder struct {
	remote
	cfg LlamaCppConfig
	// actualDim is the vector width observed from the live model,
	// learned on the first successful embed. It lets Dim() track the
	// real model even when the configured Dim is a placeholder.
	actualDim atomic.Int32
}

// NewLlamaCppEmbedder constructs the llama.cpp embedder, filling in
// sane defaults (localhost:8080, 30s timeout, 384-dim) for zero values.
// (gomem.New always passes cfg.Dim, so the 384 default applies only to
// direct callers.)
func NewLlamaCppEmbedder(cfg LlamaCppConfig) *LlamaCppEmbedder {
	if cfg.ServerURL == "" {
		cfg.ServerURL = "http://127.0.0.1:8080"
	}
	cfg.ServerURL = strings.TrimRight(cfg.ServerURL, "/")
	cfg.Timeout = positive(cfg.Timeout, 30*time.Second)
	cfg.Dim = positive(cfg.Dim, 384)
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.DiscardHandler)
	}
	return &LlamaCppEmbedder{
		remote: newRemote(cfg.ServerURL+"/embedding", cfg.APIKey, cfg.Timeout, cfg.MaxRetries, cfg.BatchSize, "gomem: llama.cpp embed"),
		cfg:    cfg,
	}
}

// Dim implements Embedder. It returns the configured dimension until the
// first successful embed, after which it returns the width the server
// actually produced. The value can therefore change once, and a consumer
// that snapshots it early (e.g. at construction) must re-check against the
// length of the vectors it really receives.
func (l *LlamaCppEmbedder) Dim() int {
	if d := l.actualDim.Load(); d > 0 {
		return int(d)
	}
	return l.cfg.Dim
}

// llamaReq is the request body for llama-server's /embedding endpoint.
// "content" is a string for a single text and an array of strings for a
// batch.
type llamaReq struct {
	Model   string `json:"model,omitempty"`
	Content any    `json:"content"`
}

// llamaResp is the legacy /embedding response: a single "embedding", or a
// "results" array for batches.
type llamaResp struct {
	Embedding []float32 `json:"embedding,omitempty"`
	Results   []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"results,omitempty"`
}

// llamaRespEntry is one element of the modern llama-server /embedding
// response: a top-level JSON array where each item carries its input
// index and the vector. The vector is doubly nested ("embedding": [[...]]);
// the inner slice is the actual float vector.
type llamaRespEntry struct {
	Index     int         `json:"index"`
	Embedding [][]float32 `json:"embedding"`
}

// Embed implements Embedder, batching inputs.
func (l *LlamaCppEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return l.embed(ctx, texts, func(batch []string) any {
		req := llamaReq{Model: l.cfg.Model, Content: batch}
		if len(batch) == 1 {
			req.Content = batch[0]
		}
		return req
	}, l.parse)
}

// parse decodes a response for n texts and records the server's width.
func (l *LlamaCppEmbedder) parse(raw []byte, n int) ([][]float32, error) {
	raw = bytes.TrimSpace(raw)
	out := make([][]float32, n)
	if len(raw) > 0 && raw[0] == '[' {
		var entries []llamaRespEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, err
		}
		for i, e := range entries {
			pos := i
			if e.Index >= 0 && e.Index < n {
				pos = e.Index
			}
			if pos < n && len(e.Embedding) > 0 {
				out[pos] = e.Embedding[0]
			}
		}
	} else {
		var lr llamaResp
		if err := json.Unmarshal(raw, &lr); err != nil {
			return nil, err
		}
		if len(lr.Results) > 0 {
			for i, r := range lr.Results {
				if i < n {
					out[i] = r.Embedding
				}
			}
		} else if lr.Embedding != nil {
			out[0] = lr.Embedding
		}
	}
	if i := slices.IndexFunc(out, func(v []float32) bool { return len(v) > 0 }); i >= 0 {
		if w := len(out[i]); l.actualDim.CompareAndSwap(0, int32(w)) && w != l.cfg.Dim {
			l.cfg.Logger.Warn("embedding server dimension differs from configured Dim; using the server's",
				"configured", l.cfg.Dim, "actual", w)
		}
	}
	return out, nil
}
