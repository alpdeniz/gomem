// Package stats gathers cardinality, storage, and health metrics
// from the database.
package stats

import (
	"context"
	"os"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
)

// Data is the collected statistics snapshot.
type Data struct {
	TotalItems   int
	ByKind       map[string]int
	ByTier       map[int]int
	DBSizeBytes  int64
	WALSizeBytes int64
	LastVacuumAt *time.Time
}

// Service computes statistics.
type Service struct {
	d *db.DB
}

// NewService wires a stats service.
func NewService(d *db.DB) *Service {
	return &Service{d: d}
}

// Get collects the current statistics.
func (s *Service) Get(ctx context.Context) (*Data, error) {
	byKind, err := countBy[string](ctx, s.d, "kind")
	if err != nil {
		return nil, err
	}
	byTier, err := countBy[int](ctx, s.d, "tier")
	if err != nil {
		return nil, err
	}
	data := &Data{
		ByKind:       byKind,
		ByTier:       byTier,
		DBSizeBytes:  fileSize(s.d.Path),
		WALSizeBytes: fileSize(s.d.Path + "-wal"),
		LastVacuumAt: s.d.LastVacuum(ctx),
	}
	for _, n := range byKind {
		data.TotalItems += n
	}
	return data, nil
}

// countBy counts live items per value of column.
func countBy[K comparable](ctx context.Context, d *db.DB, column string) (map[K]int, error) {
	rows, err := d.Raw.QueryContext(ctx,
		"SELECT "+column+", COUNT(*) FROM items WHERE deleted_at IS NULL GROUP BY "+column)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[K]int{}
	for rows.Next() {
		var k K
		var n int
		if err := rows.Scan(&k, &n); err != nil {
			return nil, err
		}
		out[k] = n
	}
	return out, rows.Err()
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}
