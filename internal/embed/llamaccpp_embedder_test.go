package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// decodeContent parses llama-server's "content" field, which is a string
// for a single text or an array of strings for a batch.
func decodeContent(raw json.RawMessage) (texts []string, single bool) {
	var s string
	if json.Unmarshal(raw, &s) == nil && s != "" {
		return []string{s}, true
	}
	_ = json.Unmarshal(raw, &texts)
	return texts, false
}

// fakeLLamaServer emulates llama-server's /embedding endpoint. It
// recognises both the single string "content" form and the batch
// array "content" form, and returns deterministic vectors derived from the input so
// the test can assert round-tripping.
func fakeLLamaServer(t *testing.T, dim int, statusFn func(int) int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embedding" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Content json.RawMessage `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		texts, single := decodeContent(req.Content)
		if len(texts) == 0 {
			http.Error(w, "no content", http.StatusBadRequest)
			return
		}

		if statusFn != nil {
			code := statusFn(callCounter(r))
			if code != 0 && code != http.StatusOK {
				http.Error(w, "boom", code)
				return
			}
		}

		w.Header().Set("Content-Type", "application/json")
		vecs := make([][]float32, len(texts))
		for i, txt := range texts {
			vecs[i] = deterministicVec(txt, dim)
		}
		if single {
			_ = json.NewEncoder(w).Encode(struct {
				Embedding []float32 `json:"embedding"`
			}{vecs[0]})
			return
		}
		type result struct {
			Embedding []float32 `json:"embedding"`
		}
		results := make([]result, len(vecs))
		for i, v := range vecs {
			results[i] = result{v}
		}
		_ = json.NewEncoder(w).Encode(struct {
			Results []result `json:"results"`
		}{results})
	}))
}

// callCounter tags each request with a monotonic counter (via a
// header the test server sets aside) so statusFn can script a
// sequence of responses. We approximate by using a package-level
// counter guarded by the test.
var callSeq int

func callCounter(_ *http.Request) int {
	callSeq++
	return callSeq
}

// deterministicVec builds a stable, L2-normalized vector from text so
// the fake server's output is reproducible.
func deterministicVec(text string, dim int) []float32 {
	v := HashVector(text, dim)
	return v
}

func TestLlamaCpp_SingleContent(t *testing.T) {
	srv := fakeLLamaServer(t, 8, nil)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 8})

	got, err := emb.Embed(context.Background(), []string{"hello world"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	want := deterministicVec("hello world", 8)
	if len(got) != 1 || len(got[0]) != 8 {
		t.Fatalf("unexpected shape: %d vectors of dim", len(got))
	}
	for i := range want {
		if got[0][i] != want[i] {
			t.Fatalf("dim %d: got %v want %v", i, got[0][i], want[i])
		}
	}
}

func TestLlamaCpp_BatchContentArray(t *testing.T) {
	srv := fakeLLamaServer(t, 8, nil)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 8})

	inputs := []string{"alpha", "beta", "gamma"}
	got, err := emb.Embed(context.Background(), inputs)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("got %d vectors for %d inputs", len(got), len(inputs))
	}
	for i, in := range inputs {
		want := deterministicVec(in, 8)
		for d := range want {
			if got[i][d] != want[d] {
				t.Fatalf("input %q dim %d: got %v want %v", in, d, got[i][d], want[d])
			}
		}
	}
}

func TestLlamaCpp_RetriesOn500(t *testing.T) {
	// Fail the first two calls with 500, succeed on the third.
	statusFn := func(seq int) int {
		if seq <= 2 {
			return http.StatusInternalServerError
		}
		return http.StatusOK
	}
	srv := fakeLLamaServer(t, 8, statusFn)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 8, MaxRetries: 3})

	if _, err := emb.Embed(context.Background(), []string{"retry me"}); err != nil {
		t.Fatalf("expected success after retries, got %v", err)
	}
}

func TestLlamaCpp_Dim(t *testing.T) {
	emb := NewLlamaCppEmbedder(LlamaCppConfig{Dim: 512})
	if emb.Dim() != 512 {
		t.Fatalf("Dim() = %d, want 512", emb.Dim())
	}
	// Zero dim falls back to the 384 default.
	def := NewLlamaCppEmbedder(LlamaCppConfig{})
	if def.Dim() != 384 {
		t.Fatalf("default Dim() = %d, want 384", def.Dim())
	}
}

// TestLlamaCpp_APIKeySentAsBearer confirms a configured API key is
// forwarded to the server as an "Authorization: Bearer <key>" header.
func TestLlamaCpp_APIKeySentAsBearer(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Embedding []float32 `json:"embedding"`
		}{deterministicVec("k", 8)})
	}))
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 8, APIKey: "secret-123"})
	if _, err := emb.Embed(context.Background(), []string{"k"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if gotAuth != "Bearer secret-123" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer secret-123")
	}
}

// TestLlamaCpp_NoAPIKeyNoHeader confirms no Authorization header is
// emitted when the key is left empty (an unauthenticated local server).
func TestLlamaCpp_NoAPIKeyNoHeader(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(struct {
			Embedding []float32 `json:"embedding"`
		}{deterministicVec("k", 8)})
	}))
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 8})
	if _, err := emb.Embed(context.Background(), []string{"k"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if gotAuth != "" {
		t.Fatalf("expected no Authorization header, got %q", gotAuth)
	}
}

// TestLlamaCpp_ModernArrayResponse pins the real-world llama-server
// contract that regressed: a TOP-LEVEL JSON array of {index, embedding},
// where "embedding" is doubly nested ([[...]]). It also declares a
// deliberately wrong Dim to prove the embedder self-calibrates to the
// model's true width (otherwise the dense leg is dropped downstream).
func TestLlamaCpp_ModernArrayResponse(t *testing.T) {
	const dim = 16
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embedding" {
			http.NotFound(w, r)
			return
		}
		var req struct {
			Content json.RawMessage `json:"content"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		texts, _ := decodeContent(req.Content)
		if len(texts) == 0 {
			http.Error(w, "no content", http.StatusBadRequest)
			return
		}
		type entry struct {
			Index     int         `json:"index"`
			Embedding [][]float32 `json:"embedding"`
		}
		entries := make([]entry, len(texts))
		for i, txt := range texts {
			entries[i] = entry{Index: i, Embedding: [][]float32{deterministicVec(txt, dim)}}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(entries)
	}))
	defer srv.Close()

	// Declared Dim is intentionally wrong (999) to prove self-calibration
	// overrides it with the model's real width.
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 999})
	inputs := []string{"one", "two", "three"}
	got, err := emb.Embed(context.Background(), inputs)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(got) != len(inputs) {
		t.Fatalf("got %d vectors for %d inputs", len(got), len(inputs))
	}
	for i, in := range inputs {
		want := deterministicVec(in, dim)
		if len(got[i]) != dim {
			t.Fatalf("input %q: dim %d, want %d", in, len(got[i]), dim)
		}
		for d := range want {
			if got[i][d] != want[d] {
				t.Fatalf("input %q dim %d: got %v want %v", in, d, got[i][d], want[d])
			}
		}
	}
	if emb.Dim() != dim {
		t.Fatalf("self-calibrated Dim() = %d, want %d", emb.Dim(), dim)
	}
}

// TestLlamaCpp_BatchBody pins the wire format: a batch is sent as
// "content": [..] and never as "prompts".
func TestLlamaCpp_BatchBody(t *testing.T) {
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		_ = json.NewEncoder(w).Encode([]struct {
			Index     int         `json:"index"`
			Embedding [][]float32 `json:"embedding"`
		}{{0, [][]float32{{1, 2}}}, {1, [][]float32{{3, 4}}}})
	}))
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 2})
	if _, err := emb.Embed(context.Background(), []string{"a", "b"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if got, want := string(body), `{"content":["a","b"]}`; got != want {
		t.Fatalf("body = %s, want %s", got, want)
	}
}

// rateLimitedServer answers 429 (with the given Retry-After, if any) for
// the first call and a valid single-vector response afterwards.
func rateLimitedServer(retryAfter string, calls *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			if retryAfter != "" {
				w.Header().Set("Retry-After", retryAfter)
			}
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/embedding") {
			_ = json.NewEncoder(w).Encode(map[string]any{"embedding": []float32{1, 2}})
			return
		}
		writeAPIResponse(w, 1)
	}))
}

func TestLlamaCpp_RetriesOn429(t *testing.T) {
	var calls atomic.Int32
	srv := rateLimitedServer("", &calls)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 2, MaxRetries: 2})
	if _, err := emb.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("expected success after 429, got %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestLlamaCpp_RetryAfterHonoured(t *testing.T) {
	var calls atomic.Int32
	srv := rateLimitedServer("1", &calls)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 2, MaxRetries: 2})
	start := time.Now()
	if _, err := emb.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if d := time.Since(start); d < time.Second {
		t.Fatalf("retried after %v, want >= Retry-After of 1s", d)
	}
}

func TestLlamaCpp_RetryAfterBoundedByContext(t *testing.T) {
	var calls atomic.Int32
	srv := rateLimitedServer("30", &calls)
	defer srv.Close()
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 2, MaxRetries: 2})
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := emb.Embed(ctx, []string{"x"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("took %v; Retry-After should not outlive the context", d)
	}
}

func TestLlamaCpp_DimMismatchWarnsOnce(t *testing.T) {
	srv := fakeLLamaServer(t, 8, nil)
	defer srv.Close()
	var buf bytes.Buffer
	lg := slog.New(slog.NewTextHandler(&buf, nil))
	emb := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: srv.URL, Dim: 4, Logger: lg})
	for i := 0; i < 3; i++ {
		if _, err := emb.Embed(context.Background(), []string{"x"}); err != nil {
			t.Fatalf("embed: %v", err)
		}
	}
	if n := strings.Count(buf.String(), "dimension differs"); n != 1 {
		t.Fatalf("warned %d times, want 1:\n%s", n, buf.String())
	}
	if emb.Dim() != 8 {
		t.Fatalf("Dim() = %d, want 8", emb.Dim())
	}
}
