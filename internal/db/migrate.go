package db

import (
	"context"
	"database/sql"
	_ "embed" // schema.sql
	"errors"
	"fmt"
	"strconv"
	"strings"
)

//go:embed schema.sql
var schemaSQL string

// migration is a single ordered schema patch.
type migration struct {
	version int
	name    string
	sql     string
	// fn, when set, runs after sql in the same transaction, for steps
	// SQLite cannot express idempotently in plain SQL.
	fn func(ctx context.Context, tx *sql.Tx) error
}

// migrations are applied in ascending version order.
var migrations = []migration{{
	version: 1,
	name:    "initial",
	sql:     schemaSQL,
}, {
	version: 2,
	name:    "superseded_by_index",
	sql: `CREATE INDEX IF NOT EXISTS idx_items_superseded_by
  ON items(superseded_by) WHERE superseded_by IS NOT NULL;`,
}, {
	// fts.item_id is UNINDEXED, so deleting an item's FTS row by item_id
	// scans the whole table. items.fts_rowid records the FTS rowid (kept
	// in FTS5's INTEGER PRIMARY KEY shadow tables, so VACUUM preserves it)
	// to delete by rowid instead. NULL means "unknown": writers fall back
	// to the item_id scan. idx_items_created is never used by any query.
	version: 3,
	name:    "fts_rowid",
	sql:     `DROP INDEX IF EXISTS idx_items_created;`,
	fn: func(ctx context.Context, tx *sql.Tx) error {
		var n int
		if err := tx.QueryRowContext(ctx,
			"SELECT COUNT(*) FROM pragma_table_info('items') WHERE name='fts_rowid'").Scan(&n); err != nil {
			return err
		}
		if n == 0 {
			if _, err := tx.ExecContext(ctx, "ALTER TABLE items ADD COLUMN fts_rowid INTEGER"); err != nil {
				return err
			}
		}
		_, err := tx.ExecContext(ctx,
			"UPDATE items SET fts_rowid = fts.rowid FROM fts WHERE fts.item_id = items.id AND items.fts_rowid IS NULL")
		return err
	},
}}

// Migrate brings the schema up to the latest version by applying any
// pending migrations. It is idempotent.
func (d *DB) Migrate(ctx context.Context) error {
	cur, err := d.currentVersion(ctx)
	if err != nil {
		return err
	}
	for _, m := range migrations {
		if m.version <= cur {
			continue
		}
		if err := d.applyMigration(ctx, m); err != nil {
			return err
		}
	}
	chmodQuiet(d.Path) // the WAL and SHM files exist by now
	return nil
}

func (d *DB) currentVersion(ctx context.Context) (int, error) {
	var v string
	err := d.Raw.QueryRowContext(ctx, "SELECT value FROM meta WHERE key='schema_version'").Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		// Fresh database: the meta table (created by migration 1) does
		// not exist yet, so the version lookup fails with "no such
		// table". Treat that as version 0 so all migrations apply.
		if strings.Contains(err.Error(), "no such table") {
			return 0, nil
		}
		return 0, err
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("gomem: bad schema_version %q: %w", v, err)
	}
	return n, nil
}

func (d *DB) applyMigration(ctx context.Context, m migration) error {
	tx, err := d.Raw.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(ctx, m.sql); err != nil {
		return fmt.Errorf("gomem: migration %d (%s): %w", m.version, m.name, err)
	}
	if m.fn != nil {
		if err := m.fn(ctx, tx); err != nil {
			return fmt.Errorf("gomem: migration %d (%s): %w", m.version, m.name, err)
		}
	}
	if err := setMeta(ctx, tx, "schema_version", strconv.Itoa(m.version)); err != nil {
		return err
	}
	return tx.Commit()
}
