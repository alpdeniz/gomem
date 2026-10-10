package db

import (
	"context"
	"path/filepath"
	"testing"
)

// TestMigrate_V1ToV2 opens a database that only has migration 1 applied
// and confirms the current code upgrades it in place.
func TestMigrate_V1ToV2(t *testing.T) {
	ctx := context.Background()
	d, _, err := Open(filepath.Join(t.TempDir(), "migrate.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	if err := d.applyMigration(ctx, migrations[0]); err != nil {
		t.Fatalf("apply v1: %v", err)
	}
	if v, _ := d.currentVersion(ctx); v != 1 {
		t.Fatalf("expected schema_version 1, got %d", v)
	}

	if err := d.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if v, _ := d.currentVersion(ctx); v != len(migrations) {
		t.Fatalf("expected schema_version %d, got %d", len(migrations), v)
	}
	var n int
	if err := d.Raw.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_items_superseded_by'").Scan(&n); err != nil {
		t.Fatalf("index lookup: %v", err)
	}
	if n != 1 {
		t.Fatal("expected idx_items_superseded_by to exist")
	}
}
