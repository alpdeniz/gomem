package recall

import (
	"context"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/model"
)

// testEnv bundles a migrated DB, an embedder, and a wired engine.
type testEnv struct {
	d   *db.DB
	emb embed.Embedder
	eng *Engine
}

// newTestEnv builds an env; an optional fusion mode selects the Engine's.
func newTestEnv(t *testing.T, emb embed.Embedder, fusion ...string) *testEnv {
	t.Helper()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "recall.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	cfg := testConfig()
	cfg.FusionMode = strings.Join(fusion, "")
	eng := NewEngine(d, emb, nil, nil, cfg)
	return &testEnv{d: d, emb: emb, eng: eng}
}

// testConfig is the engine configuration the tests share.
func testConfig() EngineConfig {
	return EngineConfig{
		Dim: 256, AlphaDense: 0.5, BetaSparse: 0.3, GammaSalience: 0.2, DefaultTopK: 10,
		Params: model.Params{HalfLifeDays: 30, FreqCap: 100, WConfidence: 0.2, WFrequency: 0.3, WRecency: 0.5},
	}
}

// insertItem seeds an item across items/vec/fts/tags.
func (e *testEnv) insertItem(t *testing.T, id, kind, title, body string, tags []string, conf float64) {
	t.Helper()
	ctx := context.Background()
	now := time.Now().UTC().Format(time.RFC3339)
	prov := `{"source":"run","confidence":` + f64str(conf) + `,"created_at":"` + now + `"}`
	if _, err := e.d.Raw.ExecContext(ctx,
		`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
		 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
		id, kind, title, body, `{}`, prov, now, now); err != nil {
		t.Fatalf("insert item: %v", err)
	}
	// A failing embedder degrades gracefully: the item is still
	// seeded (items/fts/tags), just without a dense vector.
	if vecs, err := e.emb.Embed(ctx, []string{title + "\n" + body}); err == nil {
		if _, ierr := e.d.Raw.ExecContext(ctx,
			"INSERT INTO vec(item_id, embedding) VALUES(?,?)", id, embed.EncodeVec(vecs[0])); ierr != nil {
			t.Fatalf("insert vec: %v", ierr)
		}
	}
	if _, err := e.d.Raw.ExecContext(ctx,
		"INSERT INTO fts(item_id, title, body) VALUES(?,?,?)", id, title, body); err != nil {
		t.Fatalf("insert fts: %v", err)
	}
	for _, tg := range tags {
		if _, err := e.d.Raw.ExecContext(ctx,
			"INSERT OR IGNORE INTO tags(item_id, tag) VALUES(?,?)", id, tg); err != nil {
			t.Fatalf("insert tag: %v", err)
		}
	}
}

func f64str(f float64) string {
	return strconv.FormatFloat(f, 'f', 3, 64)
}

func itoa(n int) string {
	return strconv.Itoa(n)
}

// TestRecall_HybridRanking exercises the fused dense+sparse+salience
// scoring end to end in every fusion mode.
func TestRecall_HybridRanking(t *testing.T) {
	for _, mode := range []string{"legacy", "minmax", "rrf"} {
		t.Run(mode, func(t *testing.T) { testHybridRanking(t, mode) })
	}
}

func testHybridRanking(t *testing.T, mode string) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256), mode)
	env.insertItem(t, "idx-1", "fact", "Composite indexes accelerate filtered database queries",
		"Aligning a composite index with WHERE and ORDER BY removes table scans and sorts.",
		[]string{"database", "index"}, 0.95)
	env.insertItem(t, "idx-2", "lesson", "Cache hot reads to eliminate redundant computation",
		"Memoizing frequently read, slowly changing results lowers latency and CPU spend.",
		[]string{"cache"}, 0.85)
	env.insertItem(t, "idx-3", "fact", "Authentication tokens should expire and rotate",
		"Short-lived bearer tokens with rotation limit the blast radius of a leak.",
		[]string{"auth"}, 0.9)

	res, err := env.eng.Run(context.Background(), Query{
		Goal: "speed up slow database query with indexes",
		TopK: 5,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("expected results")
	}
	// Scores normalized and descending.
	for i := 1; i < len(res); i++ {
		if res[i].Score > res[i-1].Score {
			t.Fatalf("not sorted desc at %d", i)
		}
	}
	for _, r := range res {
		if r.Score < 0 || r.Score > 1 {
			t.Fatalf("score %v out of [0,1]", r.Score)
		}
	}
	// The index item should be the top hit.
	if res[0].ID != "idx-1" {
		t.Fatalf("expected idx-1 on top, got %s", res[0].ID)
	}
}

// TestRecall_SupersededReappearsWhenSupersederDeleted confirms an item
// is hidden only while its superseder is live.
func TestRecall_SupersededReappearsWhenSupersederDeleted(t *testing.T) {
	ctx := context.Background()
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "old", "fact", "Composite indexes accelerate database queries",
		"Aligning a composite index with WHERE removes table scans.", nil, 0.9)
	env.insertItem(t, "new", "fact", "Covering indexes accelerate database queries",
		"A covering index answers the query without touching the table.", nil, 0.9)
	if _, err := env.d.Raw.ExecContext(ctx, "UPDATE items SET superseded_by='new' WHERE id='old'"); err != nil {
		t.Fatalf("link: %v", err)
	}
	has := func(res []Result, id string) bool {
		for _, r := range res {
			if r.ID == id {
				return true
			}
		}
		return false
	}
	q := Query{Goal: "database queries indexes", TopK: 10}

	res, err := env.eng.Run(ctx, q)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if has(res, "old") || !has(res, "new") {
		t.Fatalf("expected only the superseder while it is live, got %+v", res)
	}

	if _, err := env.d.Raw.ExecContext(ctx,
		"UPDATE items SET deleted_at=? WHERE id='new'", time.Now().UTC().Format(time.RFC3339)); err != nil {
		t.Fatalf("delete: %v", err)
	}
	res, err = env.eng.Run(ctx, q)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !has(res, "old") {
		t.Fatalf("expected the replaced item to reappear, got %+v", res)
	}
}

// TestRecall_EmptyDB covers the both-legs-empty case.
func TestRecall_EmptyDB(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	res, err := env.eng.Run(context.Background(), Query{Goal: "nothing here", TopK: 10})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 0 {
		t.Fatalf("expected empty, got %d", len(res))
	}
}

// TestRecall_GracefulDegrade confirms a failing embedmer leaves the
// sparse leg intact.
func TestRecall_GracefulDegrade(t *testing.T) {
	env := newTestEnv(t, &failEmbedder{dim: 256})
	env.insertItem(t, "deg-1", "fact", "Database query optimization via indexing",
		"Indexes aligned with predicates cut scan cost dramatically.",
		[]string{"database"}, 0.9)

	res, err := env.eng.Run(context.Background(), Query{
		Goal: "database query optimization indexing",
		TopK: 5,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("expected sparse-leg results despite the embedder failing")
	}
}

type failEmbedder struct{ dim int }

func (f *failEmbedder) Dim() int { return f.dim }
func (f *failEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return nil, context.DeadlineExceeded
}

type countEmbedder struct {
	embed.Embedder
	calls int
}

func (c *countEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	c.calls++
	return c.Embedder.Embed(ctx, texts)
}

// TestRecall_GoalEmbeddingCached confirms a repeated goal is embedded once.
func TestRecall_GoalEmbeddingCached(t *testing.T) {
	emb := &countEmbedder{Embedder: embed.NewOfflineEmbedder(256)}
	env := newTestEnv(t, emb)
	env.insertItem(t, "gc-1", "fact", "Database indexing", "Indexes cut scan cost.", nil, 0.9)
	base := emb.calls // seeding embeds the item
	for i := 0; i < 3; i++ {
		if _, err := env.eng.Run(context.Background(), Query{Goal: "database indexing", TopK: 5}); err != nil {
			t.Fatalf("run: %v", err)
		}
	}
	if got := emb.calls - base; got != 1 {
		t.Fatalf("goal embedded %d times, want 1", got)
	}
}

// TestRecall_DedupNearDuplicates confirms near-identical candidates
// are collapsed when a Jaccard threshold is supplied.
func TestRecall_DedupNearDuplicates(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "nd-1", "fact", "Optimize the database query with an index",
		"Adding an index aligned to the predicate removes the full scan.", []string{"db"}, 0.9)
	env.insertItem(t, "nd-2", "fact", "Optimize the database query with an index",
		"Adding an index aligned to the predicate removes the full scan.", []string{"db"}, 0.85)
	env.insertItem(t, "nd-3", "fact", "Optimize the database query with an index",
		"Adding an index aligned to the predicate removes the full scan.", []string{"db"}, 0.8)
	env.insertItem(t, "nd-distinct", "fact", "Bake a sourdough bread with a long cold ferment",
		"A slow refrigerated rise develops flavor and crumb structure.", []string{"bread"}, 0.9)

	q := Query{Goal: "optimize the database query with an index", TopK: 10}
	base, err := env.eng.Run(context.Background(), q)
	if err != nil {
		t.Fatalf("baseline run: %v", err)
	}
	baseDupes := 0
	for _, r := range base {
		if strings.HasPrefix(r.ID, "nd-") && r.ID != "nd-distinct" {
			baseDupes++
		}
	}
	if baseDupes < 2 {
		t.Fatalf("expected the near-duplicates to surface without dedup, saw %d", baseDupes)
	}

	q.DedupJaccard = 0.5
	deduped, err := env.eng.Run(context.Background(), q)
	if err != nil {
		t.Fatalf("dedup run: %v", err)
	}
	dedupDupes := 0
	for _, r := range deduped {
		if strings.HasPrefix(r.ID, "nd-") && r.ID != "nd-distinct" {
			dedupDupes++
		}
	}
	if dedupDupes >= baseDupes {
		t.Fatalf("expected dedup to reduce near-duplicates (%d -> %d)", baseDupes, dedupDupes)
	}
}

// TestRecall_TokenBudget confirms results are trimmed to fit a token
// budget when one is supplied.
func TestRecall_TokenBudget(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	for i := 0; i < 6; i++ {
		env.insertItem(t, "tb-"+itoa(i), "fact",
			"Note "+itoa(i)+" about caching and memoization strategies",
			"Entry "+itoa(i)+" discusses memoizing results to cut redundant work and lower latency.",
			[]string{"cache"}, 0.9)
	}

	q := Query{Goal: "caching and memoization strategies", TopK: 10}
	full, err := env.eng.Run(context.Background(), q)
	if err != nil {
		t.Fatalf("full run: %v", err)
	}
	if len(full) < 4 {
		t.Fatalf("expected the seeded items to surface, got %d", len(full))
	}
	if len(full) > 0 {
		perItem := full[0].Tokens
		if perItem <= 0 {
			perItem = 1
		}
		q.Budget = perItem * 2
		budgeted, err := env.eng.Run(context.Background(), q)
		if err != nil {
			t.Fatalf("budgeted run: %v", err)
		}
		if len(budgeted) >= len(full) {
			t.Fatalf("expected the budget to trim results (%d -> %d)", len(full), len(budgeted))
		}
	}
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

func BenchmarkRecall_1K(b *testing.B)   { benchRecall(b, 1_000) }
func BenchmarkRecall_10K(b *testing.B)  { benchRecall(b, 10_000) }
func BenchmarkRecall_100K(b *testing.B) { benchRecall(b, 100_000) }

// BenchmarkRecall_100K_Memory is BenchmarkRecall_100K with the in-memory
// dense index (the first query loads it; ResetTimer excludes that).
func BenchmarkRecall_100K_Memory(b *testing.B) { benchRecall(b, 100_000, "memory") }

func benchRecall(b *testing.B, n int, denseIndex ...string) {
	b.Helper()
	ctx := context.Background()
	eng, q := newBenchEngine(b, n, denseIndex...)
	if _, err := eng.Run(ctx, q); err != nil { // warm: loads the index
		b.Fatalf("warm: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Run(ctx, q); err != nil {
			b.Fatalf("run: %v", err)
		}
	}
}

// BenchmarkDenseLeg_100K isolates the dense leg (top-30, no filters) in
// both modes, without the sparse leg and candidate loading.
func BenchmarkDenseLeg_100K(b *testing.B) {
	ctx := context.Background()
	eng, q := newBenchEngine(b, 100_000, "memory")
	qvec, err := eng.embedGoal(ctx, q.Goal)
	if err != nil {
		b.Fatalf("embed: %v", err)
	}
	where, args := filterSQL(q, "i")
	b.Run("sqlite", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := denseKNN(ctx, eng.d, qvec, 30, where, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("memory", func(b *testing.B) {
		if _, err := eng.idx.search(ctx, qvec, 30, where, args); err != nil { // load
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := eng.idx.search(ctx, qvec, 30, where, args); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkRecallParallel measures concurrent recall throughput; it only
// scales when the connection pool allows parallel readers.
func BenchmarkRecallParallel(b *testing.B) {
	ctx := context.Background()
	eng, q := newBenchEngine(b, 10_000)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := eng.Run(ctx, q); err != nil {
				b.Errorf("run: %v", err)
				return
			}
		}
	})
}

// newBenchEngine seeds n items and returns an engine plus a query.
func newBenchEngine(b *testing.B, n int, denseIndex ...string) (*Engine, Query) {
	b.Helper()
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(b.TempDir(), "bench.db"), nil)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	b.Cleanup(func() { d.Close() })
	if err := d.Migrate(ctx); err != nil {
		b.Fatalf("migrate: %v", err)
	}
	emb := embed.NewOfflineEmbedder(256)

	// Bulk-seed n items in a single transaction.
	tx, err := d.Raw.BeginTx(ctx, nil)
	if err != nil {
		b.Fatalf("begin: %v", err)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	prov := `{"source":"run","confidence":0.8,"created_at":"` + now + `"}`
	for i := 0; i < n; i++ {
		id := "b-" + itoa(i)
		title := "benchmark item " + itoa(i) + " database query index"
		body := "body for benchmark item " + itoa(i) + " covering query optimization"
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
			 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
			id, "fact", title, body, `{}`, prov, now, now); err != nil {
			b.Fatalf("seed item %d: %v", i, err)
		}
		vecs, err := emb.Embed(ctx, []string{title + "\n" + body})
		if err != nil {
			b.Fatalf("seed embed %d: %v", i, err)
		}
		if _, err := tx.ExecContext(ctx,
			"INSERT INTO vec(item_id, embedding) VALUES(?,?)", id, embed.EncodeVec(vecs[0])); err != nil {
			b.Fatalf("seed vec %d: %v", i, err)
		}
		if _, err := tx.ExecContext(ctx,
			"INSERT INTO fts(item_id, title, body) VALUES(?,?,?)", id, title, body); err != nil {
			b.Fatalf("seed fts %d: %v", i, err)
		}
	}
	if err := tx.Commit(); err != nil {
		b.Fatalf("commit: %v", err)
	}

	cfg := testConfig()
	if len(denseIndex) > 0 {
		cfg.DenseIndex = denseIndex[0]
	}
	eng := NewEngine(d, emb, nil, nil, cfg)
	b.Cleanup(eng.Close)
	return eng, Query{Goal: "optimize database query performance with indexes", TopK: 10}
}

// TestRecall_QueryHalfLifeOverridesConfig pins that Query.HalfLifeDays
// reaches the recency term: the same aged item scores higher under a long
// half-life than a short one.
func TestRecall_QueryHalfLifeOverridesConfig(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "old", "fact", "Composite indexes accelerate queries", "Align the index with WHERE.", nil, 0.9)
	old := time.Now().UTC().AddDate(0, 0, -60).Format(time.RFC3339)
	if _, err := env.d.Raw.Exec("UPDATE items SET created_at=?, updated_at=?", old, old); err != nil {
		t.Fatal(err)
	}
	score := func(halfLife float64) float64 {
		t.Helper()
		res, err := env.eng.Run(context.Background(), Query{Goal: "composite indexes", TopK: 1, HalfLifeDays: halfLife})
		if err != nil || len(res) != 1 {
			t.Fatalf("run: %v (%d results)", err, len(res))
		}
		return res[0].Score
	}
	if short, long := score(1), score(3650); !(long > short) {
		t.Fatalf("half-life ignored: score %v (1d) vs %v (3650d)", short, long)
	}
}
