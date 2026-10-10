package recall

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/embed"
)

// fnEmbedder embeds each text with fn, so a test controls cosine
// similarities exactly. fail makes every Embed return an error.
type fnEmbedder struct {
	dim  int
	fn   func(text string) []float32
	fail bool
}

func (f *fnEmbedder) Dim() int { return f.dim }

func (f *fnEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if f.fail {
		return nil, errors.New("embedder down")
	}
	out := make([][]float32, len(texts))
	for i, tx := range texts {
		out[i] = f.fn(tx)
	}
	return out, nil
}

const fnDim = 8

// unit returns a vector at cosine c to e0, the rest on axis 1+k%7.
func unit(c float64, k int) []float32 {
	v := make([]float32, fnDim)
	v[0] = float32(c)
	v[1+k%7] = float32(math.Sqrt(1 - c*c))
	return v
}

// identEmbedder: the goal "errx42" is e0; the item titled "errx42 handler"
// sits at cosine 0.6 to it; "decoy-N" items at 0.9 (all keyword-free).
func identEmbedder() *fnEmbedder {
	return &fnEmbedder{dim: fnDim, fn: func(text string) []float32 {
		switch {
		case text == "errx42":
			return unit(1, 0)
		case len(text) > 5 && text[:5] == "decoy":
			return unit(0.9, len(text))
		}
		return unit(0.6, 3)
	}}
}

// freezeClock pins "now" so salience, hence scores, repeat exactly.
func freezeClock(e *Engine) {
	now := time.Now()
	e.nowFn = func() time.Time { return now }
}

func seedIdentCorpus(t *testing.T, env *testEnv) {
	t.Helper()
	for i := 0; i < 12; i++ {
		env.insertItem(t, "decoy-"+itoa(i), "fact", "decoy-"+itoa(i)+" widget", "unrelated prose", nil, 0.9)
	}
	env.insertItem(t, "ident", "fact", "errx42 handler", "retries on failure", nil, 0.9)
}

// TestRun_SparseOnlyHitKeepsRealDenseSimilarity: an exact identifier match
// that the dense leg ranked outside its top hits still carries its real
// similarity (0.6 -> 0.8 normalized) rather than 0, so with legacy fusion
// it beats decoys that are only semantically close. It also holds with the
// in-memory index, which has no per-id lookup of its own.
func TestRun_SparseOnlyHitKeepsRealDenseSimilarity(t *testing.T) {
	ctx := context.Background()
	env := newTestEnv(t, identEmbedder(), "legacy")
	seedIdentCorpus(t, env)

	memCfg := env.eng.cfg
	memCfg.DenseIndex = "memory"
	memEng := NewEngine(env.d, env.emb, nil, nil, memCfg)
	t.Cleanup(memEng.Close)

	for name, eng := range map[string]*Engine{"sqlite": env.eng, "memory": memEng} {
		t.Run(name, func(t *testing.T) {
			res, err := eng.Run(ctx, Query{Goal: "errx42", TopK: 3})
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if len(res) == 0 || res[0].ID != "ident" {
				t.Fatalf("exact identifier match should rank first, got %+v", res)
			}
		})
	}
}

// Without a query vector (embedder down) the sparse-only path is skipped:
// the keyword hit is still returned.
func TestRun_SparseOnlyKeywordFallbackWithoutVector(t *testing.T) {
	emb := identEmbedder()
	env := newTestEnv(t, emb, "legacy")
	seedIdentCorpus(t, env)
	emb.fail = true
	res, err := env.eng.Run(context.Background(), Query{Goal: "errx42", TopK: 3})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "ident" {
		t.Fatalf("want only the keyword hit, got %+v", res)
	}
}

// TestSparseOnly_OnlyLegacyFusion: minmax and rrf keep their relative
// semantics (a sparse-only item is below every dense hit), so the real
// similarity is only filled in for legacy fusion.
func TestSparseOnly_OnlyLegacyFusion(t *testing.T) {
	for mode, want := range map[string]bool{"": true, "legacy": true, "minmax": false, "rrf": false} {
		e := &Engine{cfg: EngineConfig{FusionMode: mode}}
		if e.legacyFusion() != want {
			t.Fatalf("mode %q: legacyFusion = %v, want %v", mode, !want, want)
		}
	}
}

func TestDenseLookup(t *testing.T) {
	ctx := context.Background()
	env := newTestEnv(t, identEmbedder())
	seedIdentCorpus(t, env)
	// A vector of another dimension is omitted, as is a missing id.
	if _, err := env.d.Raw.ExecContext(ctx, "UPDATE vec SET embedding = ? WHERE item_id = 'decoy-1'",
		embed.EncodeVec([]float32{1, 2, 3})); err != nil {
		t.Fatal(err)
	}
	q := unit(1, 0)
	ids := []string{"ident", "decoy-0", "decoy-1", "nope"}
	// Pad past the chunk size to exercise chunking.
	for i := 0; i < denseLookupChunk+10; i++ {
		ids = append(ids, "missing-"+itoa(i))
	}
	got, err := denseLookup(ctx, env.d, q, ids)
	if err != nil {
		t.Fatalf("lookup: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 comparable vectors, got %v", got)
	}
	if math.Abs(got["ident"]-0.6) > 1e-6 || math.Abs(got["decoy-0"]-0.9) > 1e-6 {
		t.Fatalf("got %v", got)
	}
}

// Prepared and unprepared queries give the same results, and the embed
// happens once, in Prepare.
func TestPrepare_EquivalentToRun(t *testing.T) {
	ctx := context.Background()
	var embeds atomic.Int32
	base := identEmbedder()
	emb := &fnEmbedder{dim: fnDim, fn: func(text string) []float32 {
		if text == "errx42" {
			embeds.Add(1)
		}
		return base.fn(text)
	}}
	env := newTestEnv(t, emb)
	seedIdentCorpus(t, env)
	freezeClock(env.eng)
	embeds.Store(0)

	q := Query{Goal: "errx42", TopK: 5}
	prepared := env.eng.Prepare(ctx, q)
	if embeds.Load() != 1 {
		t.Fatalf("Prepare should embed the goal once, got %d", embeds.Load())
	}
	a, err := env.eng.Run(ctx, prepared)
	if err != nil {
		t.Fatal(err)
	}
	if embeds.Load() != 1 {
		t.Fatalf("Run on a prepared query must not embed again, got %d", embeds.Load())
	}
	b, err := env.eng.Run(ctx, q)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("prepared %+v != unprepared %+v", a, b)
	}
}

// Concurrent legs must give the same answer every time.
func TestRun_ConcurrentLegsDeterministic(t *testing.T) {
	ctx := context.Background()
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	for i := 0; i < 30; i++ {
		env.insertItem(t, "m-"+itoa(i), "fact", "database index "+itoa(i%5), "query planning notes "+itoa(i), nil, 0.5+float64(i%5)/10)
	}
	freezeClock(env.eng)
	q := Query{Goal: "database index query", TopK: 8}
	want, err := env.eng.Run(ctx, q)
	if err != nil || len(want) == 0 {
		t.Fatalf("run: %v %v", err, want)
	}
	for i := 0; i < 25; i++ {
		got, err := env.eng.Run(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("run %d differs:\n%+v\n%+v", i, got, want)
		}
	}
}

func TestRun_CancelledContext(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "a", "fact", "database index", "body", nil, 0.9)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := env.eng.Run(ctx, Query{Goal: "database index", TopK: 3}); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

// The legs overlap in time: wall time is the slower leg, not the sum.
func TestRunConcurrently_OverlapsAndJoins(t *testing.T) {
	const d = 150 * time.Millisecond
	var done atomic.Int32
	start := time.Now()
	runConcurrently(
		func() { time.Sleep(d); done.Add(1) },
		func() { time.Sleep(d); done.Add(1) },
	)
	el := time.Since(start)
	if done.Load() != 2 {
		t.Fatal("runConcurrently returned before both finished")
	}
	if el >= 2*d-d/4 {
		t.Fatalf("legs ran sequentially: %v for two %v legs", el, d)
	}
}

// A panic in either leg surfaces on the caller after the other leg has
// finished, so no goroutine outlives the call.
func TestRunConcurrently_PanicsAfterJoin(t *testing.T) {
	for name, panicInA := range map[string]bool{"first": true, "second": false} {
		t.Run(name, func(t *testing.T) {
			var other atomic.Bool
			slow := func() { time.Sleep(50 * time.Millisecond); other.Store(true) }
			boom := func() { panic("boom") }
			a, b := slow, boom
			if panicInA {
				a, b = boom, slow
			}
			func() {
				defer func() {
					if r := recover(); r != "boom" {
						t.Fatalf("recovered %v, want boom", r)
					}
				}()
				runConcurrently(a, b)
				t.Fatal("expected a panic")
			}()
			if !other.Load() {
				t.Fatal("panic escaped before the other leg finished")
			}
		})
	}
}
