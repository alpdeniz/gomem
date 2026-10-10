package reflect

import (
	"context"
	"testing"
	"time"
)

// TestPurgeTombstones_RemovesFTSRows checks purge removes the FTS rows of
// purged items, both those with a recorded fts_rowid and legacy ones
// without, and leaves live and not-yet-expired rows alone.
func TestPurgeTombstones_RemovesFTSRows(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	d := eng.d

	addFTS := func(id string, record bool) {
		seedItem(t, d, id, 0.5, 0)
		res, err := d.Raw.ExecContext(ctx, "INSERT INTO fts(item_id,title,body) VALUES(?,?,?)", id, "t", "b "+id)
		if err != nil {
			t.Fatal(err)
		}
		if record {
			rowid, _ := res.LastInsertId()
			if _, err := d.Raw.ExecContext(ctx, "UPDATE items SET fts_rowid=? WHERE id=?", rowid, id); err != nil {
				t.Fatal(err)
			}
		}
	}
	addFTS("old-new", true)
	addFTS("old-legacy", false)
	addFTS("recent-del", true)
	addFTS("live", true)

	old := time.Now().UTC().AddDate(0, 0, -30).Format(time.RFC3339)
	recent := time.Now().UTC().Format(time.RFC3339)
	for id, ts := range map[string]string{"old-new": old, "old-legacy": old, "recent-del": recent} {
		if _, err := d.Raw.ExecContext(ctx, "UPDATE items SET deleted_at=? WHERE id=?", ts, id); err != nil {
			t.Fatal(err)
		}
	}

	n, err := eng.purgeTombstones(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("expected 2 purged, got %d", n)
	}
	rows, err := d.Raw.QueryContext(ctx, "SELECT item_id FROM fts ORDER BY item_id")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var got []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			t.Fatal(err)
		}
		got = append(got, id)
	}
	if len(got) != 2 || got[0] != "live" || got[1] != "recent-del" {
		t.Fatalf("unexpected remaining fts rows: %v", got)
	}
}
