package remember

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/model"
)

// countingEmbedder returns constant vectors and counts Embed calls.
type countingEmbedder struct{ calls int }

func (c *countingEmbedder) Dim() int { return 16 }

func (c *countingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	c.calls++
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = make([]float32, 16)
		out[i][0] = 1
	}
	return out, nil
}

func newTestService(t *testing.T) (*Service, *countingEmbedder, *time.Time) {
	t.Helper()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "remember.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	emb := &countingEmbedder{}
	svc := NewService(d, emb, nil, nil, 0, 0)
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	svc.nowFn = func() time.Time { return now }
	return svc, emb, &now
}

func rowTimes(t *testing.T, s *Service, id string) (updated string, prov model.ProvDoc) {
	t.Helper()
	item, err := s.fetchExisting(context.Background(), s.d.Raw, id)
	if err != nil || item == nil {
		t.Fatalf("fetchExisting: %v %v", item, err)
	}
	if err := s.d.Raw.QueryRow("SELECT updated_at FROM items WHERE id=?", id).Scan(&updated); err != nil {
		t.Fatal(err)
	}
	return updated, item.Provenance
}

func TestRemember_IdempotentNoRewrite(t *testing.T) {
	ctx := context.Background()
	svc, emb, now := newTestService(t)
	in := Input{ID: "idem-1", Kind: "fact", Title: "Idempotent", Body: "same body"}

	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	updated, prov := rowTimes(t, svc, "idem-1")
	calls := emb.calls

	*now = now.Add(time.Hour)
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	updated2, prov2 := rowTimes(t, svc, "idem-1")
	if updated2 != updated {
		t.Fatalf("updated_at rewritten: %s -> %s", updated, updated2)
	}
	if !prov2.CreatedAt.Equal(prov.CreatedAt) {
		t.Fatalf("provenance created_at changed: %s -> %s", prov.CreatedAt, prov2.CreatedAt)
	}
	if emb.calls != calls {
		t.Fatalf("embedder called again: %d -> %d", calls, emb.calls)
	}
}

func TestRemember_UpdateKeepsProvenanceCreatedAt(t *testing.T) {
	ctx := context.Background()
	svc, _, now := newTestService(t)
	in := Input{ID: "upd-1", Kind: "fact", Title: "Update", Body: "first body"}

	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	updated, prov := rowTimes(t, svc, "upd-1")

	*now = now.Add(time.Hour)
	in.Body = "second body"
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	updated2, prov2 := rowTimes(t, svc, "upd-1")
	if updated2 == updated {
		t.Fatal("updated_at not advanced on content change")
	}
	if !prov2.CreatedAt.Equal(prov.CreatedAt) {
		t.Fatalf("provenance created_at changed: %s -> %s", prov.CreatedAt, prov2.CreatedAt)
	}
}

// TestRemember_DoesNotPolluteQueryCache checks that Remember keeps document
// vectors out of the query LRU (so they cannot evict goal vectors) while
// still deduplicating repeated content through the document path.
func TestRemember_DoesNotPolluteQueryCache(t *testing.T) {
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "remember.db"), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	cache := embed.NewCache("", "")
	emb := &countingEmbedder{}
	svc := NewService(d, emb, cache, nil, 0, 0)

	cache.Put("goal", []float32{1})
	for i := 0; i < 30; i++ {
		in := Input{ID: fmt.Sprintf("doc-%d", i), Kind: "fact", Title: fmt.Sprintf("Doc %d", i), Body: fmt.Sprintf("distinct body %d", i)}
		ps, err := svc.Prepare(ctx, []Input{in})
		if err != nil {
			t.Fatal(err)
		}
		text := ps[0].in.EmbedText
		if _, ok := cache.Get(text); ok {
			t.Fatalf("document %d vector leaked into the query cache", i)
		}
		if _, ok := cache.GetDocument(text); !ok {
			t.Fatalf("document %d vector not cached on the document path", i)
		}
	}
	if _, ok := cache.Get("goal"); !ok {
		t.Fatal("goal vector lost")
	}
	calls := emb.calls
	if _, err := svc.Prepare(ctx, []Input{{ID: "doc-0", Kind: "fact", Title: "Doc 0", Body: "distinct body 0"}}); err != nil {
		t.Fatal(err)
	}
	if emb.calls != calls {
		t.Fatalf("repeated content re-embedded: %d -> %d", calls, emb.calls)
	}
}
