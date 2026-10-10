package db

import (
	"context"
	"time"
)

// FreelistRatio returns the fraction of database pages that are free.
func (d *DB) FreelistRatio(ctx context.Context) (float64, error) {
	var free, total int64
	if err := d.Raw.QueryRowContext(ctx, "PRAGMA freelist_count").Scan(&free); err != nil {
		return 0, err
	}
	if err := d.Raw.QueryRowContext(ctx, "PRAGMA page_count").Scan(&total); err != nil {
		return 0, err
	}
	if total == 0 {
		return 0, nil
	}
	return float64(free) / float64(total), nil
}

// Vacuum recompacts the database (VACUUM) and refreshes planner
// statistics (ANALYZE), recording the completion time in meta. It
// needs roughly the database's size in free disk space while it runs,
// and the caller must exclude concurrent writers.
func (d *DB) Vacuum(ctx context.Context) error {
	if _, err := d.Raw.ExecContext(ctx, "VACUUM"); err != nil {
		return err
	}
	if _, err := d.Raw.ExecContext(ctx, "ANALYZE"); err != nil {
		return err
	}
	return setMeta(ctx, d.Raw, "last_vacuum", time.Now().UTC().Format(time.RFC3339))
}
