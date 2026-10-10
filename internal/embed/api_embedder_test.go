package embed

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// writeAPIResponse emits a well-formed /embeddings response whose data
// entries are indexed in input order; each embedding's first element encodes
// the input ordinal it belongs to (the remainder are zero).
func writeAPIResponse(w http.ResponseWriter, n int) {
	data := make([]map[string]any, 0, n)
	for i := 0; i < n; i++ {
		data = append(data, map[string]any{
			"index":     i,
			"embedding": []float32{float32(i), 0, 0, 0},
		})
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
}

// TestAPIEmbedder_BatchSplit confirms inputs exceeding BatchSize are sent
// as multiple requests, each carrying at most BatchSize inputs.
func TestAPIEmbedder_BatchSplit(t *testing.T) {
	var mu sync.Mutex
	requests := [][]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		mu.Lock()
		requests = append(requests, req.Input)
		mu.Unlock()
		writeAPIResponse(w, len(req.Input))
	}))
	defer srv.Close()

	a := NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4, BatchSize: 2})
	texts := []string{"one", "two", "three", "four", "five"}
	if _, err := a.Embed(context.Background(), texts); err != nil {
		t.Fatalf("embed: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 3 { // ceil(5/2)
		t.Fatalf("expected 3 batched requests, got %d", len(requests))
	}
	for i, batch := range requests {
		if len(batch) > 2 {
			t.Fatalf("request %d carried %d inputs (> BatchSize 2)", i, len(batch))
		}
	}
}

// TestAPIEmbedder_RetryPolicy confirms transient 5xx responses are retried
// while permanent 4xx responses fail fast.
func TestAPIEmbedder_RetryPolicy(t *testing.T) {
	// 5xx then 200 -> retried, ultimately succeeds.
	var mu sync.Mutex
	calls := 0
	srv5xx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls++
		first := calls == 1
		mu.Unlock()
		if first {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		writeAPIResponse(w, len(req.Input))
	}))
	defer srv5xx.Close()
	a := NewAPIEmbedder(APIConfig{APIURL: srv5xx.URL, Dim: 4, MaxRetries: 2})
	if _, err := a.Embed(context.Background(), []string{"hello"}); err != nil {
		t.Fatalf("expected the retry to eventually succeed, got: %v", err)
	}
	mu.Lock()
	if calls != 2 {
		t.Fatalf("expected 2 attempts (1 fail + 1 success), got %d", calls)
	}
	mu.Unlock()

	// 4xx -> not retried, fails immediately.
	var mu2 sync.Mutex
	calls2 := 0
	srv4xx := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu2.Lock()
		calls2++
		mu2.Unlock()
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv4xx.Close()
	b := NewAPIEmbedder(APIConfig{APIURL: srv4xx.URL, Dim: 4, MaxRetries: 3})
	if _, err := b.Embed(context.Background(), []string{"hello"}); err == nil {
		t.Fatal("expected a 4xx to fail the embed")
	}
	mu2.Lock()
	if calls2 != 1 {
		t.Fatalf("expected exactly 1 attempt for a non-retryable 4xx, got %d", calls2)
	}
	mu2.Unlock()
}

// TestAPIEmbedder_IndexSlotPlacement confirms out-of-order response
// entries land at the slot their index designates.
func TestAPIEmbedder_IndexSlotPlacement(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		n := len(req.Input)
		data := make([]map[string]any, 0, n)
		// Respond in a scrambled order; each entry's embedding[0] encodes
		// the input ordinal it belongs to.
		for _, idx := range []int{n - 1, 0, 1} {
			if idx < 0 || idx >= n {
				continue
			}
			data = append(data, map[string]any{
				"index":     idx,
				"embedding": []float32{float32(idx), 0, 0, 0},
			})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	defer srv.Close()

	a := NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4, BatchSize: 8})
	out, err := a.Embed(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("expected 3 vectors, got %d", len(out))
	}
	for i := range out {
		if len(out[i]) != 4 {
			t.Fatalf("vector %d has dim %d, want 4", i, len(out[i]))
		}
		if out[i][0] != float32(i) {
			t.Fatalf("slot %d holds ordinal %v, want %d", i, out[i][0], i)
		}
	}
}

func TestAPIEmbedder_RetriesOn429AndHonoursRetryAfter(t *testing.T) {
	var calls atomic.Int32
	srv := rateLimitedServer("1", &calls)
	defer srv.Close()
	a := NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4, MaxRetries: 2})
	start := time.Now()
	if _, err := a.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("expected success after 429, got %v", err)
	}
	if calls.Load() != 2 || time.Since(start) < time.Second {
		t.Fatalf("calls=%d elapsed=%v, want 2 calls and >= 1s", calls.Load(), time.Since(start))
	}
}

func TestAPIEmbedder_RetryAfterBoundedByContext(t *testing.T) {
	var calls atomic.Int32
	srv := rateLimitedServer("30", &calls)
	defer srv.Close()
	a := NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4, MaxRetries: 2})
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := a.Embed(ctx, []string{"x"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}
}

func TestAPIEmbedder_SendDimensions(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		writeAPIResponse(w, 1)
	}))
	defer srv.Close()

	a := NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4})
	if _, err := a.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if !strings.Contains(body, `"dimensions":4`) {
		t.Fatalf("default body %s should include dimensions", body)
	}

	off := false
	a = NewAPIEmbedder(APIConfig{APIURL: srv.URL, Dim: 4, SendDimensions: &off})
	if _, err := a.Embed(context.Background(), []string{"x"}); err != nil {
		t.Fatalf("embed: %v", err)
	}
	if strings.Contains(body, "dimensions") {
		t.Fatalf("body %s should omit dimensions", body)
	}
}
