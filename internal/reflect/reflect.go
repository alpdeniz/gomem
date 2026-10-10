// Package reflect implements housekeeping: it demotes idle items through
// the tiers, evicts items idle past the age limit and the lowest-salience
// ones over capacity, purges tombstones, vacuums, and checkpoints the WAL.
package reflect

import (
	"context"
	"database/sql"
	"log/slog"
	"sort"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/model"
)

// Trigger names why a housekeeping pass is requested.
type Trigger string

// Trigger values.
const (
	TriggerRunEnd         Trigger = "run_end"
	TriggerPeriodic       Trigger = "periodic"
	TriggerBudgetPressure Trigger = "budget_pressure"
)

// Report summarizes a housekeeping pass.
type Report struct {
	ItemsDecayed   int
	ItemsPromoted  int
	ItemsCompacted int
	ItemsEvicted   int
	ItemsPurged    int
	DurationMs     int
}

// EngineConfig tunes the reflect engine.
type EngineConfig struct {
	MaxItems        int
	EvictionAgeDays int
	model.Params    // salience, which orders capacity eviction; HalfLifeDays also sets tier decay
}

// Engine runs housekeeping.
type Engine struct {
	d     *db.DB
	lg    *slog.Logger
	cfg   EngineConfig
	nowFn func() time.Time
}

// NewEngine wires a reflect engine.
func NewEngine(d *db.DB, lg *slog.Logger, cfg EngineConfig) *Engine {
	if lg == nil {
		lg = slog.New(slog.DiscardHandler)
	}
	return &Engine{d: d, lg: lg, cfg: cfg, nowFn: time.Now}
}

// salient pairs an active item with its computed salience scalar
// and derived age, ordering capacity eviction.
type salient struct {
	id  string
	sal float64
	age time.Duration
}

// vacuumFreeRatio is the freelist share of the file above which a
// periodic pass vacuums.
const vacuumFreeRatio = 0.25

// Run performs a housekeeping pass. Every trigger demotes aged items,
// evicts items idle longer than EvictionAgeDays, evicts the
// lowest-salience items while more than MaxItems remain, and checkpoints
// the WAL. Idleness is measured from the last access (creation, if never
// accessed) and nothing else; salience only orders the capacity victims.
// See applyAging for the exact rules.
// TriggerPeriodic also purges stale tombstones, and vacuums when over a
// quarter of the file is free pages. TriggerBudgetPressure purges and
// always vacuums. Any other trigger is treated as TriggerRunEnd.
//
// VACUUM rewrites the file, so it temporarily needs about the database's
// size in extra free disk space. The caller must hold the write lock.
func (e *Engine) Run(ctx context.Context, trigger Trigger) (*Report, error) {
	start := time.Now()
	e.lg.Info("reflect", "trigger", string(trigger))
	purge := trigger == TriggerPeriodic || trigger == TriggerBudgetPressure
	report := &Report{}
	now := e.nowFn()

	// Aging and capacity eviction share one transaction, so a failed
	// statement rolls back the whole pass.
	tx, err := e.d.Raw.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	// Age-based demotion (hot→warm→cold) and eviction of idle items.
	// Both run in SQL on the last-access timestamp, so no row is loaded.
	demoted, agedEvicted, err := e.applyAging(ctx, tx, now)
	if err != nil {
		return nil, err
	}
	report.ItemsDecayed += demoted
	report.ItemsEvicted += agedEvicted

	evicted, err := e.evictOverCapacity(ctx, tx, now)
	if err != nil {
		return nil, err
	}
	report.ItemsEvicted += evicted
	if err := tx.Commit(); err != nil {
		return nil, err
	}

	// Purge tombstones older than the retention window (hard delete).
	if purge {
		purged, err := e.purgeTombstones(ctx)
		if err != nil {
			return nil, err
		}
		report.ItemsPurged += purged
		e.maybeVacuum(ctx, trigger)
	}

	// Fold the WAL back into the main file.
	_ = e.d.Checkpoint(ctx)

	report.DurationMs = int(time.Since(start).Milliseconds())
	return report, nil
}

// maybeVacuum vacuums per the trigger's rule. The pass's changes are
// already committed, so a failure is logged rather than returned.
func (e *Engine) maybeVacuum(ctx context.Context, trigger Trigger) {
	if trigger != TriggerBudgetPressure {
		ratio, err := e.d.FreelistRatio(ctx)
		if err != nil || ratio <= vacuumFreeRatio {
			return
		}
	}
	if err := e.d.Vacuum(ctx); err != nil {
		e.lg.Warn("vacuum failed", "err", err)
	}
}

// lastTouchSQL is an item's reference time as Unix seconds: its last
// access, else its creation, else its updated_at. unixepoch yields NULL
// for an unparseable value, so a damaged last_access falls through to
// created_at and a row with no usable timestamp has a NULL reference,
// which no age comparison matches (it is never aged or evicted).
const lastTouchSQL = "COALESCE(unixepoch(last_access), unixepoch(created_at), unixepoch(updated_at))"

// beforeUnix returns the Unix-second cutoff such that a whole-second
// timestamp ts satisfies ts < cutoff exactly when now-ts > d.
func beforeUnix(now time.Time, d time.Duration) int64 {
	t := now.Add(-d)
	c := t.Unix()
	if t.Nanosecond() > 0 {
		c++
	}
	return c
}

// applyAging runs the time-based rules in SQL and returns how many rows
// each changed. An item's idle time is now minus its last access (its
// creation if it was never accessed); confidence, access count and
// salience play no part.
//
//   - Tier: idle > 4 half lives → cold (0); idle > 2 half lives → warm
//     (1); never raised. The result depends on idle time alone, so a
//     second pass changes nothing.
//   - Eviction: idle > EvictionAgeDays tombstones the item, whatever
//     its tier, confidence or lifetime access count. A Touch resets the
//     clock.
//
// Capacity eviction (evictOverCapacity) is separate.
func (e *Engine) applyAging(ctx context.Context, tx *sql.Tx, now time.Time) (demoted, evicted int, err error) {
	halfLife := daysOr(e.cfg.HalfLifeDays, 30)
	evictionAge := daysOr(float64(e.cfg.EvictionAgeDays), 90)
	q := db.NewSeq(ctx, tx)
	// One statement lowers each row to the tier its idle time warrants;
	// the WHERE clause admits only rows that actually change.
	cold, warm := beforeUnix(now, 4*halfLife), beforeUnix(now, 2*halfLife)
	demoted = int(q.Exec(`UPDATE items SET tier = CASE WHEN `+lastTouchSQL+` < ? THEN 0 ELSE 1 END
		WHERE deleted_at IS NULL AND ((tier > 0 AND `+lastTouchSQL+` < ?) OR (tier > 1 AND `+lastTouchSQL+` < ?))`,
		cold, cold, warm))
	evicted = int(q.Exec("UPDATE items SET deleted_at=? WHERE deleted_at IS NULL AND "+lastTouchSQL+" < ?",
		model.Stamp(now), beforeUnix(now, evictionAge)))
	return demoted, evicted, q.Err()
}

// daysOr is d days as a Duration, or def days when d is not positive.
func daysOr(d float64, def float64) time.Duration {
	if d <= 0 {
		d = def
	}
	return time.Duration(d * 24 * float64(time.Hour))
}

// evictOverCapacity tombstones the excess lowest-salience live items,
// ties broken by older idle time, then id. Rows are loaded (through tx, so
// they reflect the pass's own uncommitted aging) only when the live count
// actually exceeds MaxItems.
func (e *Engine) evictOverCapacity(ctx context.Context, tx *sql.Tx, now time.Time) (int, error) {
	if e.cfg.MaxItems <= 0 {
		return 0, nil
	}
	var live int
	if err := tx.QueryRowContext(ctx, "SELECT COUNT(*) FROM items WHERE deleted_at IS NULL").Scan(&live); err != nil {
		return 0, err
	}
	excess := live - e.cfg.MaxItems
	if excess <= 0 {
		return 0, nil
	}
	rows, err := tx.QueryContext(ctx,
		"SELECT id, provenance, access_count, last_access, created_at, updated_at FROM items WHERE deleted_at IS NULL")
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	var victims []salient
	for rows.Next() {
		var (
			s                    salient
			prov                 []byte
			count                int
			lastAccess           sql.NullString
			createdAt, updatedAt string
		)
		if err := rows.Scan(&s.id, &prov, &count, &lastAccess, &createdAt, &updatedAt); err != nil {
			return 0, err
		}
		u := model.NewUsage(model.ParseProv(prov).Confidence, count, createdAt, updatedAt, lastAccess, now)
		s.sal, s.age = e.cfg.Of(u, now), now.Sub(u.Ref())
		victims = append(victims, s)
	}
	if err := rows.Err(); err != nil {
		return 0, err
	}
	rows.Close() // release the connection before writing on the same tx
	sort.Slice(victims, func(i, j int) bool {
		a, b := victims[i], victims[j]
		if a.sal != b.sal {
			return a.sal < b.sal
		}
		if a.age != b.age {
			return a.age > b.age
		}
		return a.id < b.id
	})
	q := db.NewSeq(ctx, tx)
	stamp := model.Stamp(now)
	evicted := 0
	for _, s := range victims[:min(excess, len(victims))] {
		evicted += int(q.Exec("UPDATE items SET deleted_at=? WHERE id=? AND deleted_at IS NULL", stamp, s.id))
	}
	return evicted, q.Err()
}

// purgeTombstones hard-deletes tombstones older than the retention window.
func (e *Engine) purgeTombstones(ctx context.Context) (int, error) {
	cutoff := model.Stamp(e.nowFn().AddDate(0, 0, -7))
	tx, err := e.d.Raw.BeginTx(ctx, nil)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	const purged = "deleted_at IS NOT NULL AND deleted_at <= ?"
	q := db.NewSeq(ctx, tx)
	// Items superseded by a row about to be purged lose their link first;
	// superseded_by has no ON DELETE action and would block the DELETE.
	q.Exec("UPDATE items SET superseded_by=NULL WHERE superseded_by IN (SELECT id FROM items WHERE "+purged+")", cutoff)
	// The fts virtual table cannot carry a working ON DELETE CASCADE
	// (FTS5 does not parse REFERENCES in column defs), so its rows are
	// removed explicitly for the items about to be hard-deleted; vec and
	// tags cascade on their own via their regular-table foreign keys.
	// Delete by rowid (item_id is UNINDEXED: matching on it scans the
	// whole table). Legacy rows without an fts_rowid take the item_id
	// scan, run only when such rows are actually being purged.
	q.Exec("DELETE FROM fts WHERE rowid IN (SELECT fts_rowid FROM items WHERE "+purged+" AND fts_rowid IS NOT NULL)", cutoff)
	legacy := "SELECT id FROM items WHERE " + purged + " AND fts_rowid IS NULL"
	var hasLegacy bool
	if err := tx.QueryRowContext(ctx, "SELECT EXISTS("+legacy+")", cutoff).Scan(&hasLegacy); err != nil {
		return 0, err
	}
	if hasLegacy {
		q.Exec("DELETE FROM fts WHERE item_id IN ("+legacy+")", cutoff)
	}
	n := q.Exec("DELETE FROM items WHERE "+purged, cutoff)
	if err := q.Err(); err != nil {
		return 0, err
	}
	return int(n), tx.Commit()
}
