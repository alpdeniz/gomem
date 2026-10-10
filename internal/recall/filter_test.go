package recall

import (
	"context"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/embed"
)

// TestRecall_FilterDoesNotStarve confirms filters are applied inside the
// legs: a single matching item ranked far below the fetch window is found.
func TestRecall_FilterDoesNotStarve(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	for i := 0; i < 40; i++ {
		env.insertItem(t, "l-"+itoa(i), "lesson", "Database query index tuning "+itoa(i),
			"Tune the database query with an index.", nil, 0.9)
	}
	env.insertItem(t, "skill-1", "skill", "Gardening", "Prune roses in late winter.", nil, 0.9)

	res, err := env.eng.Run(context.Background(), Query{
		Goal: "database query index tuning", Kinds: []string{"skill"}, TopK: 10,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "skill-1" {
		t.Fatalf("expected only skill-1, got %+v", res)
	}
}

// TestRecall_TagsCaseInsensitive confirms query tags are normalized like
// stored tags.
func TestRecall_TagsCaseInsensitive(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "t-1", "fact", "Database indexing", "Indexes cut scan cost.", []string{"db"}, 0.9)

	res, err := env.eng.Run(context.Background(), Query{
		Goal: "database indexing", Tags: []string{" DB "}, TopK: 5,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "t-1" {
		t.Fatalf("expected t-1, got %+v", res)
	}
}

// TestRecall_TombstonesDoNotConsumeFetchWindow confirms deleted rows that
// still sit in vec/fts do not crowd out live matches.
func TestRecall_TombstonesDoNotConsumeFetchWindow(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	now := time.Now().UTC().Format(time.RFC3339)
	for i := 0; i < 40; i++ {
		id := "dead-" + itoa(i)
		env.insertItem(t, id, "fact", "Database query index tuning "+itoa(i),
			"Tune the database query with an index.", nil, 0.9)
		if _, err := env.d.Raw.Exec("UPDATE items SET deleted_at=? WHERE id=?", now, id); err != nil {
			t.Fatalf("tombstone: %v", err)
		}
	}
	env.insertItem(t, "live", "fact", "Database", "Weak match.", nil, 0.9)

	res, err := env.eng.Run(context.Background(), Query{Goal: "database query index tuning", TopK: 1})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "live" {
		t.Fatalf("expected the live item, got %+v", res)
	}
}

// TestRecall_MinConfidenceMalformedProvenance pins the fallback: provenance
// that is not valid JSON counts as confidence 1.
func TestRecall_MinConfidenceMalformedProvenance(t *testing.T) {
	env := newTestEnv(t, embed.NewOfflineEmbedder(256))
	env.insertItem(t, "bad", "fact", "Database indexing", "Indexes cut scan cost.", nil, 0.1)
	env.insertItem(t, "low", "fact", "Database indexing too", "Indexes cut scan cost.", nil, 0.1)
	if _, err := env.d.Raw.Exec("UPDATE items SET provenance='not json' WHERE id='bad'"); err != nil {
		t.Fatalf("corrupt: %v", err)
	}

	res, err := env.eng.Run(context.Background(), Query{
		Goal: "database indexing", MinConfidence: 0.9, TopK: 5,
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "bad" {
		t.Fatalf("expected only bad, got %+v", res)
	}
}

// TestRecall_NonASCIISparseLeg confirms non-ASCII queries reach the FTS
// index (the embedder fails here, so only the sparse leg can answer).
func TestRecall_NonASCIISparseLeg(t *testing.T) {
	env := newTestEnv(t, &failEmbedder{dim: 256})
	env.insertItem(t, "tr", "fact", "Veritabanı indeksi", "Çözüm önerisi sorguları hızlandırır.", nil, 0.9)
	env.insertItem(t, "ja", "fact", "データベース 索引", "検索を高速化する。", nil, 0.9)
	env.insertItem(t, "en", "fact", "Gardening", "Prune roses in late winter.", nil, 0.9)

	for goal, want := range map[string]string{"veritabanı": "tr", "cozum": "tr", "索引": "ja"} {
		res, err := env.eng.Run(context.Background(), Query{Goal: goal, TopK: 5})
		if err != nil {
			t.Fatalf("run %q: %v", goal, err)
		}
		if len(res) != 1 || res[0].ID != want {
			t.Fatalf("goal %q: expected only %s, got %+v", goal, want, res)
		}
	}
}

// TestRecall_SubGoalAddsSparseTerms confirms SubGoal tokens widen the
// sparse query (the embedder fails, so only the sparse leg answers).
func TestRecall_SubGoalAddsSparseTerms(t *testing.T) {
	env := newTestEnv(t, &failEmbedder{dim: 256})
	env.insertItem(t, "a", "fact", "Postgres indexing", "Indexes cut scan cost.", nil, 0.9)
	env.insertItem(t, "b", "fact", "Redis eviction", "Eviction policies bound memory.", nil, 0.9)

	res, err := env.eng.Run(context.Background(), Query{Goal: "postgres", TopK: 5})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 1 || res[0].ID != "a" {
		t.Fatalf("goal only: expected a, got %+v", res)
	}
	res, err = env.eng.Run(context.Background(), Query{Goal: "postgres", SubGoal: "redis", TopK: 5})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(res) != 2 {
		t.Fatalf("with subgoal: expected a and b, got %+v", res)
	}
}
