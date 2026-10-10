package db

import (
	"bytes"
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestOpen_PragmasOnEveryConnection(t *testing.T) {
	ctx := context.Background()
	d, _, err := Open(filepath.Join(t.TempDir(), "pragmas.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	d.Raw.SetMaxOpenConns(3)
	for i := 0; i < 3; i++ {
		conn, err := d.Raw.Conn(ctx)
		if err != nil {
			t.Fatalf("conn %d: %v", i, err)
		}
		defer conn.Close()
		// Hold all three open at once so each is a distinct connection.
		var fk, bt int
		if err := conn.QueryRowContext(ctx, "PRAGMA foreign_keys").Scan(&fk); err != nil || fk != 1 {
			t.Fatalf("conn %d foreign_keys = %d, err %v", i, fk, err)
		}
		if err := conn.QueryRowContext(ctx, "PRAGMA busy_timeout").Scan(&bt); err != nil || bt != 5000 {
			t.Fatalf("conn %d busy_timeout = %d, err %v", i, bt, err)
		}
	}
}

func TestOpen_PathWithSpace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "with space", "my db.db")
	d, corrupt, err := Open(path, nil)
	if err != nil || corrupt {
		t.Fatalf("open: corrupt=%v err=%v", corrupt, err)
	}
	defer d.Close()
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
}

func TestCorrupt_RecoveredDBEnforcesForeignKeys(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "corrupt.db")
	// A non-SQLite file larger than the 100-byte header.
	if err := os.WriteFile(path, bytes.Repeat([]byte("not a database "), 100), 0o600); err != nil {
		t.Fatal(err)
	}
	d, corrupt, err := Open(path, nil)
	if err != nil || !corrupt {
		t.Fatalf("open: corrupt=%v err=%v", corrupt, err)
	}
	defer d.Close()
	if err := d.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	if _, err := d.Raw.ExecContext(ctx,
		`INSERT INTO items(id, kind, title, provenance, created_at, updated_at)
		 VALUES('a','fact','t','{}','2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')`); err != nil {
		t.Fatalf("insert item: %v", err)
	}
	if _, err := d.Raw.ExecContext(ctx, "INSERT INTO tags(item_id, tag) VALUES('a','x')"); err != nil {
		t.Fatalf("insert tag: %v", err)
	}
	if _, err := d.Raw.ExecContext(ctx, "DELETE FROM items WHERE id='a'"); err != nil {
		t.Fatalf("delete item: %v", err)
	}
	var n int
	if err := d.Raw.QueryRowContext(ctx, "SELECT COUNT(*) FROM tags").Scan(&n); err != nil || n != 0 {
		t.Fatalf("tag rows after cascade = %d, err %v", n, err)
	}
}

func TestBackupAndRecreate_KeepsUncheckpointedWAL(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "wal.db")
	d, _, err := Open(path, nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	// One connection with autocheckpoint off, so commits stay in the WAL.
	d.Raw.SetMaxOpenConns(1)
	if _, err := d.Raw.ExecContext(ctx, "PRAGMA wal_autocheckpoint = 0"); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Raw.ExecContext(ctx, "CREATE TABLE t(v INTEGER)"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if _, err := d.Raw.ExecContext(ctx, "INSERT INTO t VALUES(?)", i); err != nil {
			t.Fatal(err)
		}
	}
	// A second connection keeps the database open, so closing d.Raw does
	// not run the final checkpoint (as when a corrupt file cannot be
	// checkpointed): the rows exist only in the WAL.
	holder, err := openRaw(path)
	if err != nil {
		t.Fatal(err)
	}
	holder.SetMaxOpenConns(1)
	if err := holder.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(path + "-wal"); err != nil || fi.Size() == 0 {
		t.Fatalf("expected a non-empty WAL before recovery, stat err %v", err)
	}

	if err := d.backupAndRecreate(); err != nil {
		t.Fatalf("backupAndRecreate: %v", err)
	}
	defer d.Close()

	backups, _ := filepath.Glob(path + ".corrupt-*")
	var backup string
	for _, b := range backups {
		if !strings.HasSuffix(b, "-wal") && !strings.HasSuffix(b, "-shm") {
			backup = b
		}
	}
	if backup == "" {
		t.Fatalf("no backup file among %v", backups)
	}
	if _, err := os.Stat(backup + "-wal"); err != nil {
		t.Fatalf("backup WAL missing: %v", err)
	}
	if runtime.GOOS != "windows" {
		for _, p := range []string{backup, backup + "-wal", backup + "-shm"} {
			if fi, err := os.Stat(p); err == nil && fi.Mode().Perm() != 0o600 {
				t.Errorf("%s mode = %o, want 600", filepath.Base(p), fi.Mode().Perm())
			}
		}
	}

	// The backup opens as a complete database, committed rows included.
	bk, err := sql.Open("sqlite", backup)
	if err != nil {
		t.Fatal(err)
	}
	defer bk.Close()
	var n int
	if err := bk.QueryRowContext(ctx, "SELECT COUNT(*) FROM t").Scan(&n); err != nil || n != 5 {
		t.Fatalf("backup rows = %d, err %v; want 5", n, err)
	}

	// The fresh database is empty and unaffected by the old WAL/SHM.
	if err := d.Raw.QueryRowContext(ctx, "SELECT COUNT(*) FROM sqlite_master").Scan(&n); err != nil || n != 0 {
		t.Fatalf("new db objects = %d, err %v; want empty", n, err)
	}
	var ic string
	if err := d.Raw.QueryRowContext(ctx, "PRAGMA quick_check").Scan(&ic); err != nil || ic != "ok" {
		t.Fatalf("new db quick_check = %q, err %v", ic, err)
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{path + "-wal", path + "-shm"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s lingers after close (err %v)", filepath.Base(p), err)
		}
	}
}

func TestOpen_FilePermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permissions")
	}
	path := filepath.Join(t.TempDir(), "perm.db")
	d, _, err := Open(path, nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	for _, p := range []string{path, path + "-wal", path + "-shm"} {
		fi, err := os.Stat(p)
		if err != nil {
			t.Fatalf("stat %s: %v", p, err)
		}
		if mode := fi.Mode().Perm(); mode != 0o600 {
			t.Fatalf("%s mode = %o, want 600", filepath.Base(p), mode)
		}
	}
}
