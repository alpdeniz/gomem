package remember

import (
	"context"
	"database/sql"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/alpdeniz/gomem/internal/db"
)

func ftsCount(t testing.TB, d *sql.DB, where string, args ...any) int {
	t.Helper()
	var n int
	if err := d.QueryRow("SELECT COUNT(*) FROM fts "+where, args...).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func ftsRowidOf(t testing.TB, d *sql.DB, id string) sql.NullInt64 {
	t.Helper()
	var r sql.NullInt64
	if err := d.QueryRow("SELECT fts_rowid FROM items WHERE id=?", id).Scan(&r); err != nil {
		t.Fatal(err)
	}
	return r
}

// TestRemember_FTSRowidTracked checks that every write path records the FTS
// rowid, replaces the FTS row in place, and never leaves an orphan.
func TestRemember_FTSRowidTracked(t *testing.T) {
	ctx := context.Background()
	svc, _, _ := newTestService(t)
	raw := svc.d.Raw

	in := Input{ID: "f-1", Kind: "fact", Title: "Alpha", Body: "original body"}
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	r1 := ftsRowidOf(t, raw, "f-1")
	if !r1.Valid {
		t.Fatal("fts_rowid not recorded on insert")
	}
	var itemID string
	if err := raw.QueryRow("SELECT item_id FROM fts WHERE rowid=?", r1.Int64).Scan(&itemID); err != nil || itemID != "f-1" {
		t.Fatalf("fts rowid %d does not point at f-1: %q %v", r1.Int64, itemID, err)
	}

	// Update: same single row, new content searchable, old content gone.
	in.Body = "completely replaced zebra"
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	if r2 := ftsRowidOf(t, raw, "f-1"); r2 != r1 {
		t.Fatalf("fts_rowid changed on update: %v -> %v", r1, r2)
	}
	if n := ftsCount(t, raw, "WHERE item_id='f-1'"); n != 1 {
		t.Fatalf("expected 1 fts row after update, got %d", n)
	}
	if n := ftsCount(t, raw, "WHERE fts MATCH 'zebra'"); n != 1 {
		t.Fatalf("new body not indexed: %d", n)
	}
	if n := ftsCount(t, raw, "WHERE fts MATCH 'original'"); n != 0 {
		t.Fatalf("old body still indexed: %d", n)
	}

	// Revive after a soft delete: still exactly one row.
	if _, err := raw.Exec("UPDATE items SET deleted_at='2026-01-01T00:00:00Z' WHERE id='f-1'"); err != nil {
		t.Fatal(err)
	}
	in.Body = "revived body"
	if _, err := svc.Remember(ctx, in); err != nil {
		t.Fatal(err)
	}
	if n := ftsCount(t, raw, "WHERE item_id='f-1'"); n != 1 {
		t.Fatalf("expected 1 fts row after revive, got %d", n)
	}

	// A second item gets a distinct rowid; total rows == items.
	if _, err := svc.Remember(ctx, Input{ID: "f-2", Kind: "fact", Title: "Beta", Body: "b"}); err != nil {
		t.Fatal(err)
	}
	if r := ftsRowidOf(t, raw, "f-2"); !r.Valid || r == r1 {
		t.Fatalf("f-2 rowid %v collides with f-1 %v", r, r1)
	}
	if n := ftsCount(t, raw, ""); n != 2 {
		t.Fatalf("expected 2 fts rows, got %d", n)
	}
}

// TestRemember_LegacyRowWithoutFTSRowid covers rows written by raw SQL (or
// before migration 3): fts_rowid is NULL, the item_id fallback replaces the
// FTS row, and the rowid is recorded for next time.
func TestRemember_LegacyRowWithoutFTSRowid(t *testing.T) {
	ctx := context.Background()
	svc, _, _ := newTestService(t)
	raw := svc.d.Raw

	if _, err := raw.Exec(`INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
		VALUES('leg-1','fact','Legacy','old legacy body','{}',2,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}
	if _, err := raw.Exec("INSERT INTO fts(item_id,title,body) VALUES('leg-1','Legacy','old legacy body')"); err != nil {
		t.Fatal(err)
	}
	if r := ftsRowidOf(t, raw, "leg-1"); r.Valid {
		t.Fatal("raw insert should leave fts_rowid NULL")
	}

	if _, err := svc.Remember(ctx, Input{ID: "leg-1", Kind: "fact", Title: "Legacy", Body: "new legacy body"}); err != nil {
		t.Fatal(err)
	}
	if n := ftsCount(t, raw, "WHERE item_id='leg-1'"); n != 1 {
		t.Fatalf("legacy update left %d fts rows", n)
	}
	if n := ftsCount(t, raw, "WHERE fts MATCH 'old'"); n != 0 {
		t.Fatal("stale legacy FTS row not removed")
	}
	if r := ftsRowidOf(t, raw, "leg-1"); !r.Valid {
		t.Fatal("fts_rowid not recorded after legacy update")
	}
}

// TestRememberMany_FTSNoOrphans re-remembers a batch and checks the FTS
// table stays one row per item.
func TestRememberMany_FTSNoOrphans(t *testing.T) {
	ctx := context.Background()
	svc, _, _ := newTestService(t)
	for round := 0; round < 3; round++ {
		for i := 0; i < 20; i++ {
			in := Input{ID: fmt.Sprintf("m-%d", i), Kind: "fact", Title: "T", Body: fmt.Sprintf("round %d item %d", round, i)}
			if _, err := svc.Remember(ctx, in); err != nil {
				t.Fatal(err)
			}
		}
	}
	if n := ftsCount(t, svc.d.Raw, ""); n != 20 {
		t.Fatalf("expected 20 fts rows, got %d", n)
	}
	var orphans int
	if err := svc.d.Raw.QueryRow(
		"SELECT COUNT(*) FROM fts WHERE item_id NOT IN (SELECT id FROM items)").Scan(&orphans); err != nil || orphans != 0 {
		t.Fatalf("orphans=%d err=%v", orphans, err)
	}
}

func benchRememberAt(b *testing.B, size int) {
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(b.TempDir(), "bench.db"), nil)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { d.Close() })
	if err := d.Migrate(ctx); err != nil {
		b.Fatal(err)
	}
	svc := NewService(d, &countingEmbedder{}, nil, nil, 0, 0)

	// Seed in one transaction (item + fts rows with fts_rowid recorded,
	// as Remember would), then time steady-state writes.
	tx, err := d.Raw.BeginTx(ctx, nil)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < size; i++ {
		id := fmt.Sprintf("seed-%d", i)
		body := fmt.Sprintf("seed body number %d with some filler words", i)
		if _, err := tx.Exec(`INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
			VALUES(?,'fact','seed',?,'{}',2,'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')`, id, body); err != nil {
			b.Fatal(err)
		}
		res, err := tx.Exec("INSERT INTO fts(item_id,title,body) VALUES(?,?,?)", id, "seed", body)
		if err != nil {
			b.Fatal(err)
		}
		rowid, _ := res.LastInsertId()
		if _, err := tx.Exec("UPDATE items SET fts_rowid=? WHERE id=?", rowid, id); err != nil {
			b.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Alternate update-existing and insert-new to hit both paths.
		id := fmt.Sprintf("seed-%d", i%size)
		if i%2 == 1 {
			id = fmt.Sprintf("new-%d", i)
		}
		if _, err := svc.Remember(ctx, Input{ID: id, Kind: "fact", Title: "bench", Body: fmt.Sprintf("bench body %d", i)}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRemember_1K(b *testing.B)  { benchRememberAt(b, 1_000) }
func BenchmarkRemember_10K(b *testing.B) { benchRememberAt(b, 10_000) }
