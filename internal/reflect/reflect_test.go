package reflect

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/model"
)

func newReflectEnv(t *testing.T, maxItems, evictionAgeDays int) *Engine {
	t.Helper()
	d, _, err := db.Open(filepath.Join(t.TempDir(), "reflect.db"), nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { d.Close() })
	if err := d.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	return NewEngine(d, nil, EngineConfig{
		MaxItems:        maxItems,
		EvictionAgeDays: evictionAgeDays,
		Params:          model.Params{HalfLifeDays: 30, FreqCap: 100, WConfidence: 0.2, WFrequency: 0.3, WRecency: 0.5},
	})
}

func seedItem(tb testing.TB, d *db.DB, id string, conf float64, ageDays int) {
	tb.Helper()
	ctx := context.Background()
	now := time.Now().UTC()
	created := now.AddDate(0, 0, -ageDays).Format(time.RFC3339)
	prov := `{"source":"run","confidence":` + confStr(conf) + `,"created_at":"` + created + `"}`
	if _, err := d.Raw.ExecContext(ctx,
		`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
		 VALUES(?,?,?,?,?, ?, 2, ?, ?, 0)`,
		id, "fact", "reflect item "+id, "body "+id, `{}`, prov, created, created); err != nil {
		tb.Fatalf("seed %s: %v", id, err)
	}
}

func confStr(f float64) string {
	return time.Duration(0).String()[:0] + ff(f)
}

func ff(f float64) string {
	// Minimal fixed-point rendering (avoids strconv in this package).
	if f == 0 {
		return "0"
	}
	ip := int(f)
	frac := int((f - float64(ip)) * 1000)
	s := ii(ip)
	if frac > 0 {
		s += "." + ii(frac)
	}
	return s
}

func ii(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b []byte
	for n > 0 {
		b = append(b, byte('0'+n%10))
		n /= 10
	}
	if neg {
		b = append(b, '-')
	}
	out := make([]byte, len(b))
	for i := range b {
		out[i] = b[len(b)-1-i]
	}
	return string(out)
}

// TestReflect_EvictsOverCapacity confirms low-salience items are
// tombstoned when the live count breaches the cap.
func TestReflect_EvictsOverCapacity(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 3, 90)
	d := eng.d
	for i := 0; i < 5; i++ {
		seedItem(t, d, "cap-"+ii(i), 0.8, 0)
	}
	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if report.ItemsEvicted < 2 {
		t.Fatalf("expected >=2 evictions, got %+v", report)
	}
	var live int
	if err := d.Raw.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM items WHERE deleted_at IS NULL").Scan(&live); err != nil {
		t.Fatalf("count: %v", err)
	}
	if live > 3 {
		t.Fatalf("expected <=3 live items, got %d", live)
	}
}

// TestReflect_AgedEviction confirms sufficiently stale, low-salience
// items are tombstoned by the aging pass.
func TestReflect_AgedEviction(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	d := eng.d
	// One fresh, one very stale (120d) low-confidence item.
	seedItem(t, d, "fresh", 0.9, 0)
	seedItem(t, d, "ancient", 0.1, 120)

	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if report.ItemsEvicted < 1 {
		t.Fatalf("expected the ancient item to be evicted, got %+v", report)
	}
	var ancientDeleted bool
	var del string
	if err := d.Raw.QueryRowContext(ctx,
		"SELECT deleted_at FROM items WHERE id='ancient'").Scan(&del); err == nil {
		ancientDeleted = del != ""
	}
	if !ancientDeleted {
		t.Fatal("expected the ancient item to be tombstoned")
	}
}

// TestReflect_PurgeWithSupersessionLink confirms a long-tombstoned
// superseder can be hard-deleted while another row still points at it.
func TestReflect_PurgeWithSupersessionLink(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	d := eng.d
	seedItem(t, d, "old", 0.9, 0)
	seedItem(t, d, "new", 0.9, 0)
	old := time.Now().UTC().AddDate(0, 0, -30).Format(time.RFC3339)
	if _, err := d.Raw.ExecContext(ctx, "UPDATE items SET superseded_by='new' WHERE id='old'"); err != nil {
		t.Fatalf("link: %v", err)
	}
	if _, err := d.Raw.ExecContext(ctx, "UPDATE items SET deleted_at=? WHERE id='new'", old); err != nil {
		t.Fatalf("tombstone: %v", err)
	}

	if _, err := eng.Run(ctx, TriggerPeriodic); err != nil {
		t.Fatalf("run: %v", err)
	}
	var n int
	if err := d.Raw.QueryRowContext(ctx, "SELECT COUNT(*) FROM items WHERE id='new'").Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 0 {
		t.Fatal("expected the superseder to be purged")
	}
	var sup *string
	if err := d.Raw.QueryRowContext(ctx, "SELECT superseded_by FROM items WHERE id='old'").Scan(&sup); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if sup != nil {
		t.Fatalf("expected superseded_by cleared, got %q", *sup)
	}
}

func lastVacuum(t *testing.T, eng *Engine) string {
	t.Helper()
	var v string
	err := eng.d.Raw.QueryRow("SELECT COALESCE((SELECT value FROM meta WHERE key='last_vacuum'),'')").Scan(&v)
	if err != nil {
		t.Fatalf("last_vacuum: %v", err)
	}
	return v
}

// TestReflect_TriggerSemantics confirms run_end does not purge, periodic
// purges but skips vacuum while the file is compact, and budget_pressure
// purges and vacuums.
func TestReflect_TriggerSemantics(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	d := eng.d
	old := time.Now().UTC().AddDate(0, 0, -30).Format(time.RFC3339)
	for _, id := range []string{"a", "b", "c"} {
		seedItem(t, d, id, 0.9, 0)
		if _, err := d.Raw.ExecContext(ctx, "UPDATE items SET deleted_at=? WHERE id=?", old, id); err != nil {
			t.Fatalf("tombstone: %v", err)
		}
	}
	purged := func(trigger Trigger) int {
		rep, err := eng.Run(ctx, trigger)
		if err != nil {
			t.Fatalf("run %s: %v", trigger, err)
		}
		return rep.ItemsPurged
	}
	if n := purged(TriggerRunEnd); n != 0 {
		t.Fatalf("run_end purged %d items", n)
	}
	if n := purged(TriggerPeriodic); n != 3 {
		t.Fatalf("periodic purged %d items, want 3", n)
	}
	if v := lastVacuum(t, eng); v != "" {
		t.Fatalf("periodic vacuumed a compact file: %q", v)
	}
	purged(TriggerBudgetPressure)
	if v := lastVacuum(t, eng); v == "" {
		t.Fatal("budget_pressure did not vacuum")
	}
}

// TestReflect_PeriodicVacuumsWhenFragmented confirms a periodic pass
// vacuums once more than a quarter of the file is free pages.
func TestReflect_PeriodicVacuumsWhenFragmented(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	d := eng.d
	big := strings.Repeat("x", 4096)
	for i := 0; i < 200; i++ {
		if _, err := d.Raw.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at)
			 VALUES(?,?,?,?,?,?,2,?,?)`,
			ii(i), "fact", "t", big, `{}`, `{"source":"run","confidence":1}`, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	if _, err := d.Raw.ExecContext(ctx, "DELETE FROM items"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if r, err := d.FreelistRatio(ctx); err != nil || r <= vacuumFreeRatio {
		t.Fatalf("setup: freelist ratio %v, err %v", r, err)
	}
	if _, err := eng.Run(ctx, TriggerPeriodic); err != nil {
		t.Fatalf("run: %v", err)
	}
	if lastVacuum(t, eng) == "" {
		t.Fatal("expected a vacuum")
	}
	if r, _ := d.FreelistRatio(ctx); r > vacuumFreeRatio {
		t.Fatalf("freelist ratio still %v after vacuum", r)
	}
}

func liveCount(t *testing.T, eng *Engine) int {
	t.Helper()
	var n int
	if err := eng.d.Raw.QueryRow("SELECT COUNT(*) FROM items WHERE deleted_at IS NULL").Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	return n
}

// TestReflect_AgingIsIdempotent confirms the tier depends on age alone,
// not on how many passes have run.
func TestReflect_AgingIsIdempotent(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 1000)
	seedItem(t, eng.d, "old", 0.8, 130) // > 4 half lives
	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	var tier int
	if err := eng.d.Raw.QueryRow("SELECT tier FROM items WHERE id='old'").Scan(&tier); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if tier != 0 || report.ItemsDecayed != 1 {
		t.Fatalf("tier=%d report=%+v, want tier 0 and 1 decayed", tier, report)
	}
	report, err = eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if report.ItemsDecayed != 0 {
		t.Fatalf("second pass decayed %d items", report.ItemsDecayed)
	}
}

// TestReflect_EvictionCountMatchesRows confirms items already evicted by
// age are neither re-picked for capacity nor double-counted.
func TestReflect_EvictionCountMatchesRows(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 5, 90)
	for i := 0; i < 6; i++ {
		seedItem(t, eng.d, "fresh-"+ii(i), 0.8, i)
	}
	for i := 0; i < 4; i++ {
		seedItem(t, eng.d, "stale-"+ii(i), 0.1, 120)
	}
	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if live := liveCount(t, eng); live != 5 || report.ItemsEvicted != 10-live {
		t.Fatalf("live=%d report=%+v, want 5 live and 5 evicted", live, report)
	}
}

// TestReflect_AgedNotLowestSalience confirms capacity eviction only
// removes what is still needed after age-based eviction.
func TestReflect_AgedNotLowestSalience(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 7, 100)
	for i := 0; i < 3; i++ {
		seedItem(t, eng.d, "aged-"+ii(i), 0.9, 110) // idle 110d > 100d: aged out regardless of salience
	}
	for i := 0; i < 3; i++ {
		seedItem(t, eng.d, "low-"+ii(i), 0.01, 90) // lowest salience, but idle 90d <= 100d: not aged out
	}
	for i := 0; i < 4; i++ {
		seedItem(t, eng.d, "fresh-"+ii(i), 0.8, 0)
	}
	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if live := liveCount(t, eng); live != 7 || report.ItemsEvicted != 3 {
		t.Fatalf("live=%d report=%+v, want 7 live and 3 evicted", live, report)
	}
}

// TestReflect_MalformedCreatedAtNotEvicted confirms a damaged
// created_at falls back to updated_at rather than a zero (ancient) age.
func TestReflect_MalformedCreatedAtNotEvicted(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	seedItem(t, eng.d, "damaged", 0.1, 120)
	now := time.Now().UTC().Format(time.RFC3339)
	if _, err := eng.d.Raw.Exec("UPDATE items SET created_at='garbage', updated_at=? WHERE id='damaged'", now); err != nil {
		t.Fatalf("damage: %v", err)
	}
	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if report.ItemsEvicted != 0 || liveCount(t, eng) != 1 {
		t.Fatalf("damaged item evicted: %+v", report)
	}
}

// setAccess records an access `idleDays` ago with the given lifetime count.
func setAccess(t *testing.T, eng *Engine, id string, idleDays, count int) {
	t.Helper()
	at := time.Now().UTC().AddDate(0, 0, -idleDays).Format(time.RFC3339)
	if _, err := eng.d.Raw.Exec("UPDATE items SET last_access=?, access_count=? WHERE id=?", at, count, id); err != nil {
		t.Fatalf("set access: %v", err)
	}
}

func isLive(t *testing.T, eng *Engine, id string) bool {
	t.Helper()
	var n int
	if err := eng.d.Raw.QueryRow("SELECT COUNT(*) FROM items WHERE id=? AND deleted_at IS NULL", id).Scan(&n); err != nil {
		t.Fatalf("live: %v", err)
	}
	return n == 1
}

// TestReflect_AgeEvictionIgnoresSalience pins the rule: an item idle
// longer than EvictionAgeDays is evicted whatever its confidence or
// lifetime access count, and anything touched within the window survives
// however old it is.
func TestReflect_AgeEvictionIgnoresSalience(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	seedItem(t, eng.d, "idle91", 1, 91)
	seedItem(t, eng.d, "idle89", 1, 89)
	seedItem(t, eng.d, "idle400-busy", 1, 400)
	setAccess(t, eng, "idle400-busy", 400, 50)
	seedItem(t, eng.d, "old-but-touched", 1, 400)
	setAccess(t, eng, "old-but-touched", 10, 1)
	seedItem(t, eng.d, "conf-1.0", 1, 91)
	seedItem(t, eng.d, "conf-0.8", 0.8, 91)
	seedItem(t, eng.d, "conf-0.8-fresh", 0.8, 89)

	report, err := eng.Run(ctx, TriggerPeriodic)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	for id, wantLive := range map[string]bool{
		"idle91": false, "idle89": true, "idle400-busy": false, "old-but-touched": true,
		"conf-1.0": false, "conf-0.8": false, "conf-0.8-fresh": true,
	} {
		if got := isLive(t, eng, id); got != wantLive {
			t.Errorf("%s live=%v, want %v", id, got, wantLive)
		}
	}
	if report.ItemsEvicted != 4 {
		t.Fatalf("evicted %d, want 4: %+v", report.ItemsEvicted, report)
	}
}

// TestReflect_EvictionClockIsLastAccess confirms a Touch-style access
// rescues an item whose creation date alone would evict it.
func TestReflect_EvictionClockIsLastAccess(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	seedItem(t, eng.d, "rescued", 1, 400)
	setAccess(t, eng, "rescued", 0, 1)
	if _, err := eng.Run(ctx, TriggerPeriodic); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !isLive(t, eng, "rescued") {
		t.Fatal("recently accessed item was evicted")
	}
	// Once the clock runs out again, it goes.
	eng.nowFn = func() time.Time { return time.Now().AddDate(0, 0, 91) }
	if _, err := eng.Run(ctx, TriggerPeriodic); err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if isLive(t, eng, "rescued") {
		t.Fatal("item idle 91 days was not evicted")
	}
}

// TestReflect_EvictionBoundary confirms "idle > EvictionAgeDays" is
// strict, to the second.
func TestReflect_EvictionBoundary(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 90)
	now := time.Date(2026, 6, 1, 12, 0, 0, 0, time.UTC)
	eng.nowFn = func() time.Time { return now }
	for id, idle := range map[string]time.Duration{
		"exact": 90 * 24 * time.Hour,
		"over":  90*24*time.Hour + time.Second,
	} {
		ts := now.Add(-idle).Format(time.RFC3339)
		if _, err := eng.d.Raw.Exec(
			`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
			 VALUES(?, 'fact', 't', 'b', '{}', '{"source":"run","confidence":1}', 2, ?, ?, 0)`, id, ts, ts); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	if _, err := eng.Run(ctx, TriggerRunEnd); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !isLive(t, eng, "exact") || isLive(t, eng, "over") {
		t.Fatalf("exact live=%v over live=%v, want true/false", isLive(t, eng, "exact"), isLive(t, eng, "over"))
	}
}

// TestReflect_TierFollowsIdleTime confirms demotion is driven by the last
// access as well, and that no capacity work happens under the cap.
func TestReflect_TierFollowsIdleTime(t *testing.T) {
	ctx := context.Background()
	eng := newReflectEnv(t, 1000, 1000)
	seedItem(t, eng.d, "cold", 1, 200)
	seedItem(t, eng.d, "warm", 1, 70) // > 2 half lives
	seedItem(t, eng.d, "hot", 1, 200)
	setAccess(t, eng, "hot", 5, 3)
	report, err := eng.Run(ctx, TriggerRunEnd)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	want := map[string]int{"cold": 0, "warm": 1, "hot": 2}
	for id, w := range want {
		var tier int
		if err := eng.d.Raw.QueryRow("SELECT tier FROM items WHERE id=?", id).Scan(&tier); err != nil {
			t.Fatalf("scan: %v", err)
		}
		if tier != w {
			t.Errorf("%s tier=%d, want %d", id, tier, w)
		}
	}
	if report.ItemsDecayed != 2 || report.ItemsEvicted != 0 {
		t.Fatalf("report=%+v, want 2 decayed 0 evicted", report)
	}
}

// BenchmarkReflectLarge is a steady-state pass over 30k live items that
// are under the cap and not yet due for aging.
func BenchmarkReflectLarge(b *testing.B) {
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(b.TempDir(), "bench-reflect-large.db"), nil)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()
	if err := d.Migrate(ctx); err != nil {
		b.Fatalf("migrate: %v", err)
	}
	const n = 30000
	tx, err := d.Raw.BeginTx(ctx, nil)
	if err != nil {
		b.Fatal(err)
	}
	st, err := tx.PrepareContext(ctx,
		`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
		 VALUES(?, 'fact', 't', 'b', '{}', ?, 2, ?, ?, 0)`)
	if err != nil {
		b.Fatal(err)
	}
	now := time.Now().UTC()
	for i := 0; i < n; i++ {
		ts := now.AddDate(0, 0, -i%20).Format(time.RFC3339)
		prov := `{"source":"run","run_id":"r","node_role":"worker","confidence":0.8,"evidence_refs":["a","b","c"],"created_at":"` + ts + `"}`
		if _, err := st.ExecContext(ctx, "r-"+ii(i), prov, ts, ts); err != nil {
			b.Fatal(err)
		}
	}
	st.Close()
	if err := tx.Commit(); err != nil {
		b.Fatal(err)
	}
	eng := NewEngine(d, nil, EngineConfig{
		MaxItems: n, EvictionAgeDays: 90,
		Params: model.Params{HalfLifeDays: 30, FreqCap: 100, WConfidence: 0.2, WFrequency: 0.3, WRecency: 0.5},
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Run(ctx, TriggerRunEnd); err != nil {
			b.Fatalf("run: %v", err)
		}
	}
}

// BenchmarkReflect sweeps a sized corpus.
func BenchmarkReflect(b *testing.B) {
	ctx := context.Background()
	d, _, err := db.Open(filepath.Join(b.TempDir(), "bench-reflect.db"), nil)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()
	if err := d.Migrate(ctx); err != nil {
		b.Fatalf("migrate: %v", err)
	}
	const n = 1000
	for i := 0; i < n; i++ {
		seedItem(b, d, "r-"+ii(i), 0.8, i%30)
	}
	eng := NewEngine(d, nil, EngineConfig{
		MaxItems:        n,
		EvictionAgeDays: 90,
		Params:          model.Params{HalfLifeDays: 30, FreqCap: 100, WConfidence: 0.2, WFrequency: 0.3, WRecency: 0.5},
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Run(ctx, TriggerPeriodic); err != nil {
			b.Fatalf("run: %v", err)
		}
	}
}
