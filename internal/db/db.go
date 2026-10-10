// Package db wraps a modernc.org/sqlite connection with WAL pragmas,
// file-permission enforcement, integrity checking, and a versioned
// migration runner.
package db

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

// DB is a thin lifecycle-managed wrapper around *sql.DB.
type DB struct {
	Raw  *sql.DB
	Path string
	lg   *slog.Logger

	mu     sync.Mutex
	closed bool
}

// openRaw opens the pool with the per-connection pragmas in the DSN, which
// modernc.org/sqlite applies to every new connection it creates.
func openRaw(path string) (*sql.DB, error) {
	q := url.Values{}
	for _, p := range []string{"journal_mode(WAL)", "synchronous(NORMAL)",
		"foreign_keys(1)", "busy_timeout(5000)"} {
		q.Add("_pragma", p)
	}
	raw, err := sql.Open("sqlite", path+"?"+q.Encode())
	if err != nil {
		return nil, err
	}
	// Writers are serialized by the caller; WAL lets readers run alongside.
	// One extra connection is reserved for the dense index, which pins it
	// to watch PRAGMA data_version.
	n := min(runtime.GOMAXPROCS(0), 8) + 1
	raw.SetMaxOpenConns(n)
	raw.SetMaxIdleConns(n)
	return raw, nil
}

// Open opens (creating if absent) the SQLite file, applies pragmas,
// enforces 0600 permissions on the file and its WAL/SHM sidecars, and runs
// a quick integrity check. It returns
// the handle plus a boolean reporting whether corruption was detected
// and repaired (backed up + recreated).
func Open(path string, lg *slog.Logger) (*DB, bool, error) {
	if lg == nil {
		lg = slog.New(slog.DiscardHandler)
	}
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, false, err
		}
	}
	raw, err := openRaw(path)
	if err != nil {
		return nil, false, err
	}
	d := &DB{Raw: raw, Path: path, lg: lg}
	corrupt := false
	if err := raw.PingContext(context.Background()); err != nil {
		// Corruption severe enough to break the pragmas (a bad header ->
		// "file is not a database", a torn page -> "disk image is
		// malformed") is caught here, ahead of the integrity check.
		if !isCorruptionErr(err) {
			raw.Close()
			return nil, false, err
		}
		corrupt = true
		lg.Warn("pragma failed on corrupt file; backing up and recreating", "err", err)
	} else {
		chmodQuiet(path)
		var ic string
		switch icErr := raw.QueryRow("PRAGMA quick_check").Scan(&ic); {
		case icErr != nil:
			// The check itself errored (a lock, an I/O fault): health is
			// unconfirmed but not proven corrupt, so proceed.
			lg.Warn("integrity check query failed; proceeding", "err", icErr)
		case ic != "ok":
			corrupt = true
			lg.Warn("integrity check failed; backing up and recreating", "result", ic)
		}
	}
	if corrupt {
		if err := d.backupAndRecreate(); err != nil {
			lg.Error("backup/recreate failed", "err", err)
			_ = d.Raw.Close()
			return nil, true, err
		}
	}
	return d, corrupt, nil
}

// isCorruptionErr reports whether a pragma/open error indicates a
// structurally damaged file (as opposed to, say, a permission fault).
func isCorruptionErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, tok := range []string{"not a database", "malformed", "disk image", "corrupt", "checksum"} {
		if strings.Contains(msg, tok) {
			return true
		}
	}
	return false
}

// backupAndRecreate moves the corrupt file aside and reopens a fresh
// database in its place (v1 tolerates the data loss). The -wal and -shm
// sidecars move with it under the same suffix: committed transactions that
// were never checkpointed live only in the WAL, and SQLite opens
// "<backup>-wal" next to "<backup>", so the backup stays complete.
func (d *DB) backupAndRecreate() error {
	d.Raw.Close()
	stamp := time.Now().UTC().Format("20060102T150405")
	backup := fmt.Sprintf("%s.corrupt-%s", d.Path, stamp)
	if err := os.Rename(d.Path, backup); err != nil {
		return err
	}
	for _, suffix := range []string{"-wal", "-shm"} {
		err := os.Rename(d.Path+suffix, backup+suffix)
		if err == nil || os.IsNotExist(err) {
			continue
		}
		// A stale sidecar left at the original name would be replayed
		// into the fresh database, so it must not stay.
		if rerr := os.Remove(d.Path + suffix); rerr != nil && !os.IsNotExist(rerr) {
			return fmt.Errorf("move %s aside: %w", d.Path+suffix, err)
		}
	}
	chmodQuiet(backup)
	raw, err := openRaw(d.Path)
	if err != nil {
		return err
	}
	d.Raw = raw
	if err := raw.PingContext(context.Background()); err != nil {
		return err
	}
	chmodQuiet(d.Path)
	return nil
}

// Checkpoint forces a WAL checkpoint (TRUNCATE) to fold the log back into
// the main file.
func (d *DB) Checkpoint(ctx context.Context) error {
	var busy, logPages, remaining int
	err := d.Raw.QueryRowContext(ctx, "PRAGMA wal_checkpoint(TRUNCATE)").Scan(&busy, &logPages, &remaining)
	if err == nil && busy != 0 {
		d.lg.Debug("wal checkpoint busy; readers active", "log_pages", logPages, "remaining", remaining)
	}
	return err
}

// Close checkpoints the WAL, optimizes, and releases the handle. Safe to
// call more than once (subsequent calls are no-ops).
func (d *DB) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	d.closed = true
	ctx := context.Background()
	d.Raw.QueryRowContext(ctx, "PRAGMA wal_checkpoint(TRUNCATE)").Scan(new(int), new(int), new(int))
	d.Raw.ExecContext(ctx, "PRAGMA optimize")
	chmodQuiet(d.Path)
	return d.Raw.Close()
}

// chmodQuiet restricts the main file and its WAL/SHM sidecars (which hold
// the same content) to 0600, ignoring files that do not exist yet.
func chmodQuiet(path string) {
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		_ = os.Chmod(p, 0o600)
	}
}
