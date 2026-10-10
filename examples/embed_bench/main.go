// Command embed_bench measures the Remember path against a live embedding
// API: for a fixed duration it inserts fresh, unique memory items (so the
// embedding cache never answers) and reports throughput and latency
// statistics for the end-to-end Remember call and the embed call within it.
//
// The embedding server comes from GOMEM_EMBED_API_URL / GOMEM_EMBED_API_KEY
// (or -url / -key). Remember silently degrades to keyword-only storage when
// the embedder fails, so every embed call is checked and failures are
// reported separately; the run exits non-zero if any occurred.
//
//	go run ./examples/embed_bench -d 10s -c 4
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/alpdeniz/gomem"
	"github.com/alpdeniz/gomem/internal/embed"
)

func main() {
	var (
		duration    = flag.Duration("d", 10*time.Second, "benchmark duration")
		concurrency = flag.Int("c", 1, "concurrent Remember workers")
		provider    = flag.String("provider", "llama.cpp", `embedding provider: "llama.cpp" or "openai"`)
		url         = flag.String("url", os.Getenv(gomem.EnvAPIURL), "embedding server URL (llama-server base URL, or /embeddings URL for openai)")
		key         = flag.String("key", os.Getenv(gomem.EnvAPIKey), "embedding API key")
		model       = flag.String("model", "", "embedding model id")
		dim         = flag.Int("dim", 512, "embedding dimension")
		words       = flag.Int("words", 40, "words per item body")
		timeout     = flag.Duration("timeout", 30*time.Second, "per-request timeout")
	)
	flag.Parse()
	if *url == "" {
		log.Fatalf("no embedding server: set %s or pass -url", gomem.EnvAPIURL)
	}
	*concurrency = max(*concurrency, 1)

	inner, err := newEmbedder(*provider, *url, *key, *model, *dim, *timeout)
	if err != nil {
		log.Fatal(err)
	}
	emb := &timedEmbedder{inner: inner}

	dir, err := os.MkdirTemp("", "gomem-embed-bench-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	cfg := gomem.DefaultConfig()
	cfg.Path = dir + "/bench.db"
	cfg.Dim = *dim
	cfg.EmbedLengthThreshold = 0
	cfg.Embedder = emb
	mem, err := gomem.New(context.Background(), cfg)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer mem.Close()

	// Warm up the connection (and let llama.cpp settle its dimension) so
	// the first measured call does not pay for TCP/TLS setup.
	if _, err := inner.Embed(context.Background(), []string{"warm up"}); err != nil {
		log.Fatalf("warm-up embed failed: %v", err)
	}

	fmt.Printf("embed_bench: provider=%s url=%s dim=%d workers=%d duration=%s words=%d\n",
		*provider, *url, *dim, *concurrency, *duration, *words)

	var (
		seq       atomic.Int64
		mu        sync.Mutex
		remembers []time.Duration
		failed    int
		wg        sync.WaitGroup
	)
	start := time.Now()
	deadline := start.Add(*duration)
	for w := range *concurrency {
		wg.Go(func() {
			rng := rand.New(rand.NewPCG(uint64(w), uint64(start.UnixNano())))
			var local []time.Duration
			var localFailed int
			for time.Now().Before(deadline) {
				item := newItem(rng, seq.Add(1), *words)
				ctx, cancel := context.WithTimeout(context.Background(), *timeout)
				t0 := time.Now()
				_, err := mem.Remember(ctx, item)
				el := time.Since(t0)
				cancel()
				if err != nil {
					localFailed++
					log.Printf("remember: %v", err)
					continue
				}
				local = append(local, el)
			}
			mu.Lock()
			remembers = append(remembers, local...)
			failed += localFailed
			mu.Unlock()
		})
	}
	wg.Wait()
	wall := time.Since(start)

	st, err := mem.Stats(context.Background())
	if err != nil {
		log.Fatalf("stats: %v", err)
	}
	embeds, embedFailures, embedErr := emb.snapshot()

	fmt.Printf("\nwall time          %s\n", wall.Round(time.Millisecond))
	fmt.Printf("items stored       %d (remember errors: %d)\n", st.TotalItems, failed)
	fmt.Printf("embed calls        %d ok, %d failed\n", len(embeds), embedFailures)
	fmt.Printf("throughput         %.2f items/s (embedded %.2f items/s)\n",
		float64(len(remembers))/wall.Seconds(), float64(len(embeds))/wall.Seconds())
	fmt.Printf("db size            %d bytes (+%d WAL)\n\n", st.DBSizeBytes, st.WALSizeBytes)

	fmt.Printf("%-10s %8s %10s %10s %10s %10s %10s %10s %10s\n",
		"latency", "n", "min", "mean", "p50", "p90", "p99", "max", "stddev")
	printRow("remember", remembers)
	printRow("embed", embeds)
	if len(remembers) > 0 && len(embeds) > 0 {
		fmt.Printf("\nstore overhead     %s mean (remember - embed)\n",
			(mean(remembers) - mean(embeds)).Round(time.Microsecond))
	}

	if embedFailures > 0 || failed > 0 {
		fmt.Printf("\nFAIL: %d embed failures (items stored keyword-only), %d remember errors; last embed error: %v\n",
			embedFailures, failed, embedErr)
		os.Exit(1)
	}
}

func newEmbedder(provider, url, key, model string, dim int, timeout time.Duration) (embed.Embedder, error) {
	switch strings.ToLower(provider) {
	case "llama.cpp", "llamacpp", "llama":
		return embed.NewLlamaCppEmbedder(embed.LlamaCppConfig{
			ServerURL: url, APIKey: key, Model: model, Dim: dim, Timeout: timeout, MaxRetries: 1,
		}), nil
	case "openai", "api":
		return embed.NewAPIEmbedder(embed.APIConfig{
			APIURL: url, APIKey: key, Model: model, Dim: dim, Timeout: timeout, MaxRetries: 1,
		}), nil
	}
	return nil, fmt.Errorf("unknown provider %q (want llama.cpp or openai)", provider)
}

// timedEmbedder records the latency of every successful embed call and
// counts failures, including wrong-shape results that Remember would reject.
type timedEmbedder struct {
	inner embed.Embedder

	mu       sync.Mutex
	ok       []time.Duration
	failures int
	lastErr  error
}

func (t *timedEmbedder) Dim() int { return t.inner.Dim() }

func (t *timedEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	t0 := time.Now()
	vecs, err := t.inner.Embed(ctx, texts)
	el := time.Since(t0)
	if err == nil && len(vecs) != len(texts) {
		err = fmt.Errorf("got %d vectors for %d texts", len(vecs), len(texts))
	}
	if err == nil {
		for _, v := range vecs {
			if len(v) != t.inner.Dim() {
				err = fmt.Errorf("got %d-dim vector, want %d (set -dim)", len(v), t.inner.Dim())
				break
			}
		}
	}
	t.mu.Lock()
	if err != nil {
		t.failures++
		t.lastErr = err
	} else {
		t.ok = append(t.ok, el)
	}
	t.mu.Unlock()
	return vecs, err
}

func (t *timedEmbedder) snapshot() ([]time.Duration, int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.ok), t.failures, t.lastErr
}

var vocab = strings.Fields(`agent memory recall embedding vector index query cache
latency throughput database schema migration retry backoff timeout planner tool
result evidence lesson skill fact template goal context window token budget
salience recency frequency confidence provenance cluster shard replica network
request response server client batch stream queue worker scheduler profile`)

// newItem builds a unique item: the sequence number and random word mix
// guarantee distinct content, so the embedding cache never short-circuits.
func newItem(rng *rand.Rand, n int64, words int) gomem.Item {
	var b strings.Builder
	fmt.Fprintf(&b, "Observation %d:", n)
	for range words {
		b.WriteByte(' ')
		b.WriteString(vocab[rng.IntN(len(vocab))])
	}
	return gomem.Item{
		Kind:  gomem.KindFact,
		Title: fmt.Sprintf("bench item %d %s %s", n, vocab[rng.IntN(len(vocab))], vocab[rng.IntN(len(vocab))]),
		Body:  b.String(),
		Tags:  []string{"bench"},
	}
}

func printRow(name string, ds []time.Duration) {
	if len(ds) == 0 {
		fmt.Printf("%-10s %8d\n", name, 0)
		return
	}
	s := slices.Clone(ds)
	slices.Sort(s)
	r := func(d time.Duration) string { return d.Round(10 * time.Microsecond).String() }
	fmt.Printf("%-10s %8d %10s %10s %10s %10s %10s %10s %10s\n", name, len(s),
		r(s[0]), r(mean(s)), r(pct(s, 50)), r(pct(s, 90)), r(pct(s, 99)), r(s[len(s)-1]), r(stddev(s)))
}

// pct returns the p-th percentile (nearest rank) of sorted durations.
func pct(sorted []time.Duration, p float64) time.Duration {
	i := int(float64(len(sorted))*p/100+0.5) - 1
	return sorted[max(min(i, len(sorted)-1), 0)]
}

func mean(ds []time.Duration) time.Duration {
	var sum time.Duration
	for _, d := range ds {
		sum += d
	}
	return sum / time.Duration(len(ds))
}

func stddev(ds []time.Duration) time.Duration {
	m := float64(mean(ds))
	var ss float64
	for _, d := range ds {
		x := float64(d) - m
		ss += x * x
	}
	return time.Duration(math.Sqrt(ss / float64(len(ds))))
}
