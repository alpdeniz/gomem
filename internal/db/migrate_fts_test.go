package db

import (
	"context"
	"path/filepath"
	"testing"
)

// TestMigrate_V2ToV3_BackfillsFTSRowid opens a database at schema version
// 2 holding items and fts rows, then checks migration 3 backfills
// items.fts_rowid, drops idx_items_created, and is safe to re-run.
func TestMigrate_V2ToV3_BackfillsFTSRowid(t *testing.T) {
	ctx := context.Background()
	d, _, err := Open(filepath.Join(t.TempDir(), "v2.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	for _, m := range migrations[:2] {
		if err := d.applyMigration(ctx, m); err != nil {
			t.Fatalf("apply v%d: %v", m.version, err)
		}
	}
	if v, _ := d.currentVersion(ctx); v != 2 {
		t.Fatalf("expected schema_version 2, got %d", v)
	}
	for _, id := range []string{"a", "b", "c"} {
		if _, err := d.Raw.ExecContext(ctx, `INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
			VALUES(?, 'fact', 't', 'b', '{}', 2, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')`, id); err != nil {
			t.Fatal(err)
		}
		if _, err := d.Raw.ExecContext(ctx, "INSERT INTO fts(item_id,title,body) VALUES(?,?,?)", id, "t", "b "+id); err != nil {
			t.Fatal(err)
		}
	}
	// An item with no FTS row stays NULL.
	if _, err := d.Raw.ExecContext(ctx, `INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
		VALUES('nofts', 'fact', 't', 'b', '{}', 2, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')`); err != nil {
		t.Fatal(err)
	}

	if err := d.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if v, _ := d.currentVersion(ctx); v != 3 {
		t.Fatalf("expected schema_version 3, got %d", v)
	}

	var mismatched, backfilled, nulls int
	if err := d.Raw.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM items i JOIN fts ON fts.rowid = i.fts_rowid WHERE fts.item_id <> i.id`).Scan(&mismatched); err != nil {
		t.Fatal(err)
	}
	if err := d.Raw.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM items i JOIN fts ON fts.rowid = i.fts_rowid WHERE fts.item_id = i.id`).Scan(&backfilled); err != nil {
		t.Fatal(err)
	}
	if err := d.Raw.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM items WHERE fts_rowid IS NULL`).Scan(&nulls); err != nil {
		t.Fatal(err)
	}
	if mismatched != 0 || backfilled != 3 || nulls != 1 {
		t.Fatalf("backfill wrong: mismatched=%d backfilled=%d nulls=%d", mismatched, backfilled, nulls)
	}

	var n int
	if err := d.Raw.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM sqlite_master WHERE name='idx_items_created'").Scan(&n); err != nil || n != 0 {
		t.Fatalf("idx_items_created should be dropped: n=%d err=%v", n, err)
	}

	// Re-applying migration 3 (column already present) must not fail.
	if err := d.applyMigration(ctx, migrations[2]); err != nil {
		t.Fatalf("re-apply v3: %v", err)
	}
}
