package remember

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
)

// toggleEmbedder fails while fail is set and otherwise returns constant
// vectors.
type toggleEmbedder struct{ fail bool }

func (e *toggleEmbedder) Dim() int { return 16 }

func (e *toggleEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	if e.fail {
		return nil, errors.New("embed down")
	}
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = make([]float32, 16)
		out[i][0] = 1
	}
	return out, nil
}

func newToggleService(t *testing.T) (*Service, *toggleEmbedder, *time.Time) {
	t.Helper()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "backfill.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	emb := &toggleEmbedder{}
	svc := NewService(d, emb, nil, nil, 0, 0)
	now := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	svc.nowFn = func() time.Time { return now }
	return svc, emb, &now
}

func countRows(t *testing.T, s *Service, table, id string) int {
	t.Helper()
	col := "item_id"
	if table == "items" {
		col = "id"
	}
	var n int
	if err := s.d.Raw.QueryRow("SELECT COUNT(*) FROM "+table+" WHERE "+col+"=?", id).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestRemember_BackfillsMissingVector(t *testing.T) {
	ctx := context.Background()
	svc, emb, now := newToggleService(t)
	in := Input{ID: "bf-1", Kind: "fact", Title: "Backfill", Body: "same body"}

	emb.fail = true
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, svc, "vec", "bf-1"); n != 0 {
		t.Fatalf("vec rows after failed embed = %d, want 0", n)
	}
	updated, prov := rowTimes(t, svc, "bf-1")

	// Still failing: the idempotent path stays a no-op.
	*now = now.Add(time.Hour)
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, svc, "vec", "bf-1"); n != 0 {
		t.Fatalf("vec rows after second failure = %d, want 0", n)
	}

	// Embedder recovers: the vector is backfilled, nothing else changes.
	emb.fail = false
	*now = now.Add(time.Hour)
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	if n := countRows(t, svc, "vec", "bf-1"); n != 1 {
		t.Fatalf("vec rows after backfill = %d, want 1", n)
	}
	if n := countRows(t, svc, "items", "bf-1"); n != 1 {
		t.Fatalf("items rows = %d, want 1", n)
	}
	if n := countRows(t, svc, "fts", "bf-1"); n != 1 {
		t.Fatalf("fts rows = %d, want 1", n)
	}
	updated2, prov2 := rowTimes(t, svc, "bf-1")
	if updated2 != updated || !prov2.CreatedAt.Equal(prov.CreatedAt) {
		t.Fatalf("backfill rewrote item state: updated %s -> %s", updated, updated2)
	}
}

func TestRemember_SameContentWithVectorStillShortCircuits(t *testing.T) {
	ctx := context.Background()
	svc, _, now := newToggleService(t)
	in := Input{ID: "bf-2", Kind: "fact", Title: "Has vec", Body: "same body"}

	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	var before []byte
	if err := svc.d.Raw.QueryRow("SELECT embedding FROM vec WHERE item_id=?", "bf-2").Scan(&before); err != nil {
		t.Fatal(err)
	}
	updated, _ := rowTimes(t, svc, "bf-2")

	// Mutate the stored vector so a rewrite would be detectable.
	marker := make([]byte, len(before))
	if _, err := svc.d.Raw.Exec("UPDATE vec SET embedding=? WHERE item_id=?", marker, "bf-2"); err != nil {
		t.Fatal(err)
	}
	*now = now.Add(time.Hour)
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	var after []byte
	if err := svc.d.Raw.QueryRow("SELECT embedding FROM vec WHERE item_id=?", "bf-2").Scan(&after); err != nil {
		t.Fatal(err)
	}
	if string(after) != string(marker) {
		t.Fatal("vector rewritten on unchanged re-remember")
	}
	if updated2, _ := rowTimes(t, svc, "bf-2"); updated2 != updated {
		t.Fatalf("updated_at rewritten: %s -> %s", updated, updated2)
	}
}

func TestRememberMany_BackfillsMissingVector(t *testing.T) {
	ctx := context.Background()
	svc, emb, _ := newToggleService(t)
	ins := []Input{
		{ID: "bm-1", Kind: "fact", Title: "One", Body: "first"},
		{ID: "bm-2", Kind: "fact", Title: "Two", Body: "second"},
	}
	run := func() {
		t.Helper()
		ps, err := svc.Prepare(ctx, ins)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := svc.CommitAll(ctx, ps); err != nil {
			t.Fatal(err)
		}
	}

	emb.fail = true
	run()
	emb.fail = false
	run()
	run() // third pass is a pure no-op
	for _, in := range ins {
		if n := countRows(t, svc, "vec", in.ID); n != 1 {
			t.Fatalf("%s: vec rows = %d, want 1", in.ID, n)
		}
		if n := countRows(t, svc, "items", in.ID); n != 1 {
			t.Fatalf("%s: items rows = %d, want 1", in.ID, n)
		}
	}
}
