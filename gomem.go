// Package gomem is a small, pure-Go, embeddable memory substrate for
// autonomous agents. It layers hybrid dense-vector + sparse-keyword
// recall, a computed (unstored) salience scalar that feeds the recall
// score and orders capacity eviction, provenance tracking, and
// per-leg degradation (keyword-only recall when the embedder is
// unavailable) over a single portable SQLite file (WAL) with an
// embedded store and no database daemon. Tier decay and age eviction
// depend only on time since last access, not on salience.
package gomem

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/model"
	"github.com/alpdeniz/gomem/internal/recall"
	refl "github.com/alpdeniz/gomem/internal/reflect"
	"github.com/alpdeniz/gomem/internal/remember"
	"github.com/alpdeniz/gomem/internal/stats"
)

// Version is the library release identifier.
const Version = "0.1.0"

// Memory is the top-level handle. Construct with New and release with
// Close.
type Memory interface {
	// Recall performs hybrid retrieval (dense + sparse + salience).
	Recall(ctx context.Context, q RecallQuery) ([]Hit, error)
	// Remember persists a new item (or updates an existing one by ID),
	// returning the item's ID: the caller's if one was given, otherwise a
	// new ULID. Re-remembering an existing ID with unchanged content is a
	// no-op, except that a vector missing from an earlier failed embed is
	// backfilled. If the embedder is unavailable the item is stored
	// keyword-only rather than failing.
	Remember(ctx context.Context, item Item) (string, error)
	// RememberMany persists items in one transaction, returning their IDs
	// in order. It is all-or-nothing: if any item is invalid or fails to
	// write, none is stored.
	RememberMany(ctx context.Context, items []Item) ([]string, error)
	// Get retrieves a single item by its ID, or ErrNotFound if it is absent
	// or tombstoned. Its Payload is the best-effort JSON decoding of the
	// stored document (structs come back as maps).
	Get(ctx context.Context, id string) (Item, error)
	// Delete tombstones the item with the given ID — a soft delete that
	// recall filters out and Reflect eventually purges — returning
	// ErrNotFound if it is absent or already deleted.
	Delete(ctx context.Context, id string) error
	// Touch records an access to the item with the given ID: it bumps its
	// access count and refreshes its last-access time, which raises its
	// salience and resets the idle clock that drives tier decay and age
	// eviction. It never restores a tier already demoted. It returns
	// ErrNotFound if the item is absent or tombstoned.
	Touch(ctx context.Context, id string) error
	// Reflect triggers housekeeping: tier decay, eviction, WAL checkpoint,
	// and, depending on the trigger, tombstone purge and vacuum (see
	// ReflectTrigger). Items idle (since last access) longer than
	// Config.EvictionAgeDays are evicted regardless of confidence or
	// access count; over Config.MaxItems the lowest computed (unstored)
	// salience is evicted first.
	Reflect(ctx context.Context, trigger ReflectTrigger) (*ReflectReport, error)
	// Stats returns cardinality, storage, and health metrics.
	Stats(ctx context.Context) (*Stats, error)
	// Close flushes the WAL and releases the handle. Idempotent.
	Close() error
}

type memory struct {
	cfg        Config
	d          *db.DB
	recaller   *recall.Engine
	rememberer *remember.Service
	reflector  *refl.Engine
	statser    *stats.Service

	// mu orders access to the store by effect: readers (Recall, Get,
	// Stats) take RLock and proceed in parallel (the connection pool holds
	// several WAL readers); writers (Remember, Delete, Touch, Reflect)
	// take Lock and are mutually exclusive. It also guards closed.
	mu     sync.RWMutex
	closed bool
}

// rlock takes the shared lock for a read and returns its release, or
// ErrClosed.
func (m *memory) rlock() (unlock func(), err error) {
	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return nil, ErrClosed
	}
	return m.mu.RUnlock, nil
}

// lock takes the exclusive lock for a write and returns its release, or
// ErrClosed.
func (m *memory) lock() (unlock func(), err error) {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, ErrClosed
	}
	return m.mu.Unlock, nil
}

// New opens (or creates) the memory database, applies migrations, and
// wires the embedder, recall, remember, reflect, and stats engines.
//
// If the on-disk file was found corrupt, it (with its -wal and -shm
// files) is moved aside as a backup and a fresh database is recreated; the
// returned Memory is usable, and the error wraps ErrCorrupt (v1 tolerates
// the data loss).
func New(ctx context.Context, cfg Config) (Memory, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	lg := cfg.Logger
	if lg == nil {
		lg = slog.New(slog.DiscardHandler)
	}
	d, corrupt, err := db.Open(cfg.Path, lg)
	if err != nil {
		return nil, err
	}
	if err := d.Migrate(ctx); err != nil {
		d.Close()
		return nil, err
	}
	emb := buildEmbedder(cfg, lg)
	cache := embed.NewCache(cfg.Embed.CacheDir, cacheNamespace(cfg, emb))
	salience := model.Params{
		HalfLifeDays: cfg.RecencyHalfLifeDays,
		FreqCap:      cfg.FreqCap,
		WConfidence:  cfg.WConfidence,
		WFrequency:   cfg.WFrequency,
		WRecency:     cfg.WRecency,
	}
	m := &memory{
		cfg: cfg,
		d:   d,
		recaller: recall.NewEngine(d, emb, cache, lg, recall.EngineConfig{
			Dim:           cfg.Dim,
			AlphaDense:    cfg.AlphaDense,
			BetaSparse:    cfg.BetaSparse,
			GammaSalience: cfg.GammaSalience,
			Params:        salience,
			DefaultTopK:   cfg.DefaultTopK,
			DenseIndex:    cfg.DenseIndex,
			FusionMode:    cfg.FusionMode,
		}),
		rememberer: remember.NewService(d, emb, cache, lg, cfg.MaxPayloadKB, cfg.EmbedLengthThreshold),
		reflector: refl.NewEngine(d, lg, refl.EngineConfig{
			MaxItems:        cfg.MaxItems,
			EvictionAgeDays: cfg.EvictionAgeDays,
			Params:          salience,
		}),
		statser: stats.NewService(d),
	}
	if corrupt {
		return m, fmt.Errorf("%w: backed up and recreated", ErrCorrupt)
	}
	return m, nil
}

// Remember persists an item atomically (item + vector + FTS + tags),
// generating a ULID when the ID is empty. It is RememberMany of one item.
func (m *memory) Remember(ctx context.Context, item Item) (string, error) {
	ids, err := m.RememberMany(ctx, []Item{item})
	if err != nil {
		return "", err
	}
	return ids[0], nil
}

// RememberMany persists items in one transaction. Validation and
// embedding (one batched call) run without the lock (the embedder can be
// slow); only the commit takes the exclusive lock, so concurrent
// Remembers/Reflects cannot interleave.
func (m *memory) RememberMany(ctx context.Context, items []Item) ([]string, error) {
	unlock, err := m.rlock()
	if err != nil {
		return nil, err
	}
	unlock()
	ins := make([]remember.Input, len(items))
	for i, item := range items {
		if ins[i], err = item.toInput(m.cfg.MaxPayloadKB); err != nil {
			return nil, err
		}
	}
	ps, err := m.rememberer.Prepare(ctx, ins)
	if err != nil {
		return nil, err
	}
	unlock, err = m.lock()
	if err != nil {
		return nil, err
	}
	defer unlock()
	if err := m.checkSupersededBy(ctx, items); err != nil {
		return nil, err
	}
	return m.recaller.IndexWrite(ctx, func() ([]string, error) {
		return m.rememberer.CommitAll(ctx, ps)
	})
}

// checkSupersededBy rejects an item whose SupersededBy differs from the
// stored link: it is output only. It reads the store, so callers hold the
// lock.
func (m *memory) checkSupersededBy(ctx context.Context, items []Item) error {
	for _, item := range items {
		if item.SupersededBy == "" {
			continue
		}
		var stored sql.NullString
		if item.ID != "" {
			err := m.d.Raw.QueryRowContext(ctx,
				"SELECT superseded_by FROM items WHERE id = ?", item.ID).Scan(&stored)
			if err != nil && !errors.Is(err, sql.ErrNoRows) {
				return err
			}
		}
		if stored.String != item.SupersededBy {
			return fmt.Errorf("%w: SupersededBy is read-only; use Supersedes", ErrInvalid)
		}
	}
	return nil
}

// Recall runs hybrid retrieval and returns shaped hits sorted by
// descending composite score. It is a read: it holds the shared lock, so
// concurrent Recalls/Stats proceed in parallel. The goal is embedded
// before the lock is taken: the embedder never touches the database, and a
// slow embed under the lock would stall every writer, and, behind a pending
// writer, every new reader.
func (m *memory) Recall(ctx context.Context, q RecallQuery) ([]Hit, error) {
	iq := m.recaller.Prepare(ctx, recall.Query{
		Goal:              q.Goal,
		SubGoal:           q.SubGoal,
		Kinds:             kindsToStrings(q.Kinds),
		MinTier:           q.MinTier,
		Tags:              q.Tags,
		MinConfidence:     q.MinConfidence,
		TopK:              positive(q.TopK, m.cfg.DefaultTopK),
		Budget:            q.Budget,
		HalfLifeDays:      positive(q.RecencyHalfLifeDays, m.cfg.RecencyHalfLifeDays),
		IncludeSuperseded: q.IncludeSuperseded,
		DedupJaccard:      positive(q.DedupJaccardThreshold, m.cfg.DedupJaccardThreshold),
	})
	unlock, err := m.rlock()
	if err != nil {
		return nil, err
	}
	defer unlock()
	results, err := m.recaller.Run(ctx, iq)
	if err != nil {
		return nil, err
	}
	hits := make([]Hit, len(results))
	for i, r := range results {
		hits[i] = Hit{
			ID:         r.ID,
			Kind:       Kind(r.Kind),
			Title:      r.Title,
			Body:       r.Body,
			Score:      r.Score,
			Provenance: publicProvenance(r.Provenance),
			Tier:       r.Tier,
			Tokens:     r.Tokens,
		}
	}
	return hits, nil
}

// Get retrieves a single item by ID. It is a read: it holds the shared
// lock, so concurrent Gets/Recalls/Stats proceed in parallel.
func (m *memory) Get(ctx context.Context, id string) (Item, error) {
	unlock, err := m.rlock()
	if err != nil {
		return Item{}, err
	}
	defer unlock()
	return m.loadItem(ctx, id)
}

// Delete tombstones an item by ID. It is a write: it holds the exclusive
// lock, serialized against Remember/Reflect/other Deletes.
func (m *memory) Delete(ctx context.Context, id string) error {
	unlock, err := m.lock()
	if err != nil {
		return err
	}
	defer unlock()
	return m.updateOne(ctx, []string{id},
		"UPDATE items SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL", model.Stamp(time.Now()), id)
}

// Touch records an access to an item by ID. It is a write: it holds the
// exclusive lock, serialized against Remember/Reflect/other Touches.
func (m *memory) Touch(ctx context.Context, id string) error {
	unlock, err := m.lock()
	if err != nil {
		return err
	}
	defer unlock()
	// Touch changes no vector; going through IndexWrite keeps its commit
	// from looking like a foreign write.
	return m.updateOne(ctx, nil,
		"UPDATE items SET access_count = access_count + 1, last_access = ? WHERE id = ? AND deleted_at IS NULL",
		model.Stamp(time.Now()), id)
}

// updateOne runs a single-row UPDATE through the dense index's write path
// (changed lists the IDs whose vector or liveness it affects), reporting
// ErrNotFound when no row matched. Callers hold the write lock.
func (m *memory) updateOne(ctx context.Context, changed []string, query string, args ...any) error {
	_, err := m.recaller.IndexWrite(ctx, func() ([]string, error) {
		res, err := m.d.Raw.ExecContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		if n, _ := res.RowsAffected(); n == 0 {
			return nil, ErrNotFound
		}
		return changed, nil
	})
	return err
}

// loadItem fetches a live item (with its tags) by ID, rebuilding the
// public Item. It reports ErrNotFound for an absent or tombstoned id.
// Callers hold the appropriate lock.
func (m *memory) loadItem(ctx context.Context, id string) (Item, error) {
	var (
		it           Item
		tier         int
		payload      []byte
		prov         []byte
		supersededBy sql.NullString
	)
	err := m.d.Raw.QueryRowContext(ctx,
		`SELECT id, kind, title, body, payload, provenance, tier,
		        (SELECT s.id FROM items s WHERE s.id = items.superseded_by AND s.deleted_at IS NULL)
		 FROM items WHERE id = ? AND deleted_at IS NULL`, id).
		Scan(&it.ID, &it.Kind, &it.Title, &it.Body, &payload, &prov, &tier, &supersededBy)
	if errors.Is(err, sql.ErrNoRows) {
		return Item{}, ErrNotFound
	}
	if err != nil {
		return Item{}, err
	}
	it.Tier = &tier
	it.SupersededBy = supersededBy.String
	it.Provenance = publicProvenance(model.ParseProv(prov))
	if len(payload) > 0 {
		var p any
		if json.Unmarshal(payload, &p) == nil {
			it.Payload = p
		}
	}
	err = m.d.Raw.QueryRowContext(ctx,
		"SELECT id FROM items WHERE superseded_by = ? AND deleted_at IS NULL ORDER BY updated_at DESC LIMIT 1", id).
		Scan(&it.Supersedes)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return Item{}, err
	}
	if it.Tags, err = db.Tags(ctx, m.d.Raw, id); err != nil {
		return Item{}, err
	}
	return it, nil
}

// Reflect runs a housekeeping pass and returns its report. It is a
// write (it mutates tiers, tombstones, and checkpoints the WAL, and may
// purge and vacuum): it holds the exclusive lock, serialized against
// Remember and other Reflects.
func (m *memory) Reflect(ctx context.Context, trigger ReflectTrigger) (*ReflectReport, error) {
	unlock, err := m.lock()
	if err != nil {
		return nil, err
	}
	defer unlock()
	rep, err := m.reflector.Run(ctx, refl.Trigger(trigger))
	m.recaller.InvalidateIndex() // eviction and purge change liveness in bulk
	if err != nil {
		return nil, err
	}
	out := ReflectReport(*rep)
	return &out, nil
}

// Stats returns the current metrics snapshot. It is a read: it holds
// the shared lock, so concurrent Stats/Recalls proceed in parallel.
func (m *memory) Stats(ctx context.Context) (*Stats, error) {
	unlock, err := m.rlock()
	if err != nil {
		return nil, err
	}
	defer unlock()
	data, err := m.statser.Get(ctx)
	if err != nil {
		return nil, err
	}
	byKind := make(map[Kind]int, len(data.ByKind))
	for k, c := range data.ByKind {
		byKind[Kind(k)] = c
	}
	return &Stats{
		TotalItems:   data.TotalItems,
		ByKind:       byKind,
		ByTier:       data.ByTier,
		DBSizeBytes:  data.DBSizeBytes,
		WALSizeBytes: data.WALSizeBytes,
		LastVacuumAt: data.LastVacuumAt,
	}, nil
}

// Close checkpoints the WAL and releases the handle. Calling it more
// than once is a no-op. It takes the exclusive lock to flip closed, so
// no in-flight read or write observes a half-closed store.
func (m *memory) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	m.mu.Unlock()
	m.recaller.Close()
	return m.d.Close()
}
