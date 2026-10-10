package db

import (
	"context"
	"database/sql"
	"time"
)

// Querier is satisfied by *sql.DB and *sql.Tx.
type Querier interface {
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// Seq runs statements against a transaction in order and remembers the
// first failure, so a sequence of writes needs one error check: after a
// failure every further Exec is skipped.
type Seq struct {
	ctx context.Context
	tx  *sql.Tx
	err error
}

// NewSeq starts a statement sequence on tx.
func NewSeq(ctx context.Context, tx *sql.Tx) *Seq { return &Seq{ctx: ctx, tx: tx} }

func (s *Seq) run(query string, args []any) (res sql.Result) {
	if s.err == nil {
		res, s.err = s.tx.ExecContext(s.ctx, query, args...)
	}
	if s.err != nil {
		return nil
	}
	return res
}

// Exec runs one statement and returns the rows it changed (0 after a
// failure; see Err).
func (s *Seq) Exec(query string, args ...any) (n int64) {
	if res := s.run(query, args); res != nil {
		n, s.err = res.RowsAffected()
	}
	return n
}

// Insert runs an INSERT and returns its rowid (0 after a failure).
func (s *Seq) Insert(query string, args ...any) (id int64) {
	if res := s.run(query, args); res != nil {
		id, s.err = res.LastInsertId()
	}
	return id
}

// Err is the first failure, if any.
func (s *Seq) Err() error { return s.err }

// Tags returns an item's tags in ascending order.
func Tags(ctx context.Context, q Querier, id string) ([]string, error) {
	rows, err := q.QueryContext(ctx, "SELECT tag FROM tags WHERE item_id = ? ORDER BY tag", id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tags []string
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		tags = append(tags, t)
	}
	return tags, rows.Err()
}

// setMeta upserts a key in the meta table.
func setMeta(ctx context.Context, x interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
}, key, value string) error {
	_, err := x.ExecContext(ctx,
		"INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", key, value)
	return err
}

// LastVacuum returns when Vacuum last completed (nil if it never has).
func (d *DB) LastVacuum(ctx context.Context) *time.Time {
	var v string
	if d.Raw.QueryRowContext(ctx, "SELECT value FROM meta WHERE key='last_vacuum'").Scan(&v) != nil {
		return nil
	}
	t, err := time.Parse(time.RFC3339, v)
	if err != nil {
		return nil
	}
	return &t
}
