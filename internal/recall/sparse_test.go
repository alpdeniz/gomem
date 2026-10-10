package recall

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
)

func TestBuildFTSQuery_DropsStopwords(t *testing.T) {
	got := buildFTSQuery("how do I fix the slow database")
	want := `"fix" OR "slow" OR "database"`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestBuildFTSQuery_KeepsMeaningfulShortTokens(t *testing.T) {
	got := buildFTSQuery("why is the Go build using C and E0502 in v2")
	want := `"Go" OR "build" OR "using" OR "C" OR "E0502" OR "v2"`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestBuildFTSQuery_AllStopwordsFallsBack(t *testing.T) {
	if got := buildFTSQuery("the"); got != `"the"` {
		t.Fatalf("got %q", got)
	}
	if got := buildFTSQuery("to be or to be"); got != `"to" OR "be" OR "or" OR "to" OR "be"` {
		t.Fatalf("got %q", got)
	}
}

// TestSparseMatch_AllStopwordQueryStillReturns checks that a query made
// only of stopwords still matches documents containing them.
func TestSparseMatch_AllStopwordQueryStillReturns(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "s1", "fact", "The plan", "the quick note", nil, 0.9)
	where, args := filterSQL(Query{}, "i")
	hits, err := sparseMatch(context.Background(), env.d, buildFTSQuery("the"), 5, where, args)
	if err != nil || len(hits) != 1 || hits[0].ID != "s1" {
		t.Fatalf("hits %v err %v", hits, err)
	}
}

// legacySparseMatch is the previous single-query implementation (FTS joined
// to items, filtered in SQL); it is the reference for the new one.
func legacySparseMatch(ctx context.Context, d *db.DB, query string, topK int, where string, args []any) ([]hit, error) {
	rows, err := d.Raw.QueryContext(ctx,
		"SELECT fts.item_id, bm25(fts) AS score FROM fts JOIN items i ON i.id = fts.item_id "+
			"WHERE fts MATCH ? AND "+where+" ORDER BY score ASC LIMIT ?",
		append(append([]any{query}, args...), topK)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []hit
	for rows.Next() {
		var h hit
		if err := rows.Scan(&h.ID, &h.Score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// TestSparseMatch_SelectiveFilterReachesPastFirstChunk: the only items
// passing the filter rank far beyond the first FTS fetch, and must still be
// found.
func TestSparseMatch_SelectiveFilterReachesPastFirstChunk(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	// 400 strong lessons outrank every skill: more than the first fetch
	// (topK*4, min 64) and than one further round.
	for i := 0; i < 400; i++ {
		env.insertItem(t, fmt.Sprintf("l%03d", i), "lesson", "cache invalidation strategy cache cache",
			"cache invalidation on write cache", nil, 0.9)
	}
	for i := 0; i < 3; i++ {
		env.insertItem(t, fmt.Sprintf("sk%d", i), "skill", "Strategy notes", "a strategy for one-off jobs", nil, 0.9)
	}
	ctx := context.Background()
	ftsQuery := buildFTSQuery("cache invalidation strategy")

	where, args := filterSQL(Query{Kinds: []string{"skill"}}, "i")
	hits, err := sparseMatch(ctx, env.d, ftsQuery, 10, where, args)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d skill hits, want 3: %v", len(hits), hits)
	}
	for _, h := range hits {
		if !strings.HasPrefix(h.ID, "sk") {
			t.Fatalf("unexpected hit %v", h)
		}
	}

	// And end to end through Engine.Run.
	res, err := env.eng.Run(ctx, Query{Goal: "cache invalidation strategy", Kinds: []string{"skill"}, TopK: 3})
	if err != nil || len(res) == 0 {
		t.Fatalf("run: %v %v", res, err)
	}
	for _, r := range res {
		if r.Kind != "skill" {
			t.Fatalf("filter violated: %+v", r)
		}
	}
}

// TestSparseMatch_MatchesLegacy compares against the single-query reference
// over a mix of filters, including ones that make several rounds necessary.
func TestSparseMatch_MatchesLegacy(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	filler := genFiller(600, 7)
	for i, it := range filler {
		kind := []string{"fact", "lesson", "skill"}[i%3]
		var tags []string
		if i%7 == 0 {
			tags = []string{"db"}
		}
		env.insertItem(t, it.id, kind, it.title, it.body, tags, 0.5+0.1*float64(i%5))
	}
	ctx := context.Background()
	if _, err := env.d.Raw.ExecContext(ctx, "UPDATE items SET deleted_at=? WHERE id IN (?,?,?)",
		time.Now().UTC().Format(time.RFC3339), filler[0].id, filler[1].id, filler[2].id); err != nil {
		t.Fatal(err)
	}
	queries := []Query{
		{},
		{Kinds: []string{"skill"}},
		{Kinds: []string{"lesson", "skill"}, Tags: []string{"db"}},
		{Tags: []string{"db"}, MinConfidence: 0.8},
		{MinTier: 3},
	}
	for _, goal := range []string{"how do I fix the slow database", "retry the failed request", "cache"} {
		ftsQuery := buildFTSQuery(goal)
		for qi, q := range queries {
			where, args := filterSQL(q, "i")
			for _, topK := range []int{5, 30} {
				want, err := legacySparseMatch(ctx, env.d, ftsQuery, topK, where, args)
				if err != nil {
					t.Fatal(err)
				}
				got, err := sparseMatch(ctx, env.d, ftsQuery, topK, where, args)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(want) {
					t.Fatalf("%q q%d k%d: got %d hits, want %d", goal, qi, topK, len(got), len(want))
				}
				for i := range got {
					if got[i].Score != want[i].Score {
						t.Fatalf("%q q%d k%d: score[%d] %v != %v", goal, qi, topK, i, got[i].Score, want[i].Score)
					}
				}
			}
		}
	}
}

// BenchmarkSparseNaturalLanguage measures the sparse leg for a natural
// language goal at 30K items; "legacy_join" is the previous implementation
// with the old (stopword-including) query, "run" is the full Engine.Run.
//
//	go test ./internal/recall -run '^$' -bench SparseNaturalLanguage
func BenchmarkSparseNaturalLanguage(b *testing.B) {
	const goal = "how do I fix the slow database"
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(b.TempDir(), "sparse.db"), nil)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { d.Close() })
	if err := d.Migrate(ctx); err != nil {
		b.Fatal(err)
	}
	emb := embed.NewOfflineEmbedder(256)
	tx, err := d.Raw.BeginTx(ctx, nil)
	if err != nil {
		b.Fatal(err)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	prov := `{"source":"run","confidence":0.8,"created_at":"` + now + `"}`
	for _, it := range genFiller(30_000, 1) {
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
			 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
			it.id, "fact", it.title, it.body, `{}`, prov, now, now); err != nil {
			b.Fatal(err)
		}
		if _, err := tx.ExecContext(ctx, "INSERT INTO vec(item_id, embedding) VALUES(?,?)",
			it.id, embed.EncodeVec(embed.HashVector(it.title+"\n"+it.body, 256))); err != nil {
			b.Fatal(err)
		}
		if _, err := tx.ExecContext(ctx, "INSERT INTO fts(item_id, title, body) VALUES(?,?,?)",
			it.id, it.title, it.body); err != nil {
			b.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		b.Fatal(err)
	}
	eng := NewEngine(d, emb, nil, nil, testConfig())
	b.Cleanup(eng.Close)

	q := Query{Goal: goal, TopK: 10}
	where, args := filterSQL(q, "i")
	const fetch = 30
	var legacyQuery string
	for i, t := range strings.Fields(goal) {
		if i > 0 {
			legacyQuery += " OR "
		}
		legacyQuery += `"` + t + `"`
	}
	b.Run("legacy_join", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := legacySparseMatch(ctx, d, legacyQuery, fetch, where, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("legacy_join_nostop", func(b *testing.B) {
		fq := buildFTSQuery(goal)
		for i := 0; i < b.N; i++ {
			if _, err := legacySparseMatch(ctx, d, fq, fetch, where, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("sparse", func(b *testing.B) {
		fq := buildFTSQuery(goal)
		for i := 0; i < b.N; i++ {
			if _, err := sparseMatch(ctx, d, fq, fetch, where, args); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("run", func(b *testing.B) {
		if _, err := eng.Run(ctx, q); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := eng.Run(ctx, q); err != nil {
				b.Fatal(err)
			}
		}
	})
}
