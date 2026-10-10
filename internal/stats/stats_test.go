package stats

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/alpdeniz/gomem/internal/db"
)

func newStatsEnv(t *testing.T) (*Service, *db.DB) {
	t.Helper()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "stats.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return NewService(d), d
}

func TestStats_CountsByKind(t *testing.T) {
	ctx := context.Background()
	svc, d := newStatsEnv(t)
	now := "2026-10-04T00:00:00Z"
	prov := `{"source":"run","confidence":0.9,"created_at":"` + now + `"}`
	seed := []struct{ id, kind string }{
		{"a", "fact"}, {"b", "fact"}, {"c", "lesson"}, {"d", "skill"},
	}
	for _, s := range seed {
		if _, err := d.Raw.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
			 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
			s.id, s.kind, "t", "b", `{}`, prov, now, now); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	data, err := svc.Get(ctx)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if data.TotalItems != 4 {
		t.Fatalf("expected 4 items, got %d", data.TotalItems)
	}
	if data.ByKind["fact"] != 2 {
		t.Fatalf("expected 2 facts, got %d", data.ByKind["fact"])
	}
	if data.ByKind["lesson"] != 1 || data.ByKind["skill"] != 1 {
		t.Fatalf("unexpected distribution: %+v", data.ByKind)
	}
}

func TestStats_ExcludesTombstones(t *testing.T) {
	ctx := context.Background()
	svc, d := newStatsEnv(t)
	now := "2026-10-04T00:00:00Z"
	prov := `{"source":"run","confidence":0.9,"created_at":"` + now + `"}`
	for _, id := range []string{"live1", "live2", "dead"} {
		if _, err := d.Raw.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
			 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
			id, "fact", "t", "b", `{}`, prov, now, now); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	// Tombstone one item.
	if _, err := d.Raw.ExecContext(ctx,
		"UPDATE items SET deleted_at=? WHERE id='dead'", now); err != nil {
		t.Fatalf("tombstone: %v", err)
	}
	data, err := svc.Get(ctx)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if data.TotalItems != 2 {
		t.Fatalf("expected 2 live items (tombstone excluded), got %d", data.TotalItems)
	}
}
