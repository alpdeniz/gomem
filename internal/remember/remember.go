package remember

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"slices"
	"time"
	"unicode/utf8"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/errs"
	"github.com/alpdeniz/gomem/internal/model"
)

// Input is the normalized item handed to the persistence service.
type Input struct {
	ID         string
	Kind       string
	Title      string
	Body       string
	Payload    []byte
	Provenance model.ProvDoc
	// Tier is nil when the caller did not choose one: Commit then keeps the
	// stored tier of a live item and gives a new (or revived) item DefaultTier.
	Tier       *int
	Tags       []string
	Supersedes string
	EmbedText  string
}

// DefaultTier is the tier of a new item whose caller gave none: hot, the
// schema default, from which Reflect decays it with idle time.
const DefaultTier = 2

// Service persists items atomically.
type Service struct {
	d            *db.DB
	emb          embed.Embedder
	cache        *embed.Cache
	lg           *slog.Logger
	maxPayloadKB int
	// minEmbedLen is the minimum EmbedText length (in runes) worth
	// embedding; shorter items are stored FTS-only. 0 disables the check.
	minEmbedLen int
	nowFn       func() time.Time
}

// NewService wires a persistence service.
func NewService(d *db.DB, emb embed.Embedder, cache *embed.Cache, lg *slog.Logger, maxPayloadKB, minEmbedLen int) *Service {
	if lg == nil {
		lg = slog.New(slog.DiscardHandler)
	}
	if cache == nil {
		cache = embed.NewCache("", "")
	}
	return &Service{d: d, emb: emb, cache: cache, lg: lg, maxPayloadKB: maxPayloadKB, minEmbedLen: minEmbedLen, nowFn: time.Now}
}

// Prepared is a normalized item with its embedding, ready to Commit. It is
// produced without touching the database.
type Prepared struct {
	in                 Input
	vec                []float32
	createdAtDefaulted bool
}

// Prepare validates and normalizes every input and embeds them in one
// call (cache hits skipped); it performs no database access, so callers
// can run it outside any lock. Any invalid input fails the whole batch.
// An embedding failure degrades to FTS-only storage.
func (s *Service) Prepare(ctx context.Context, ins []Input) ([]Prepared, error) {
	ps := make([]Prepared, len(ins))
	var missIdx []int
	var missTexts []string
	for i, in := range ins {
		createdAtDefaulted := in.Provenance.CreatedAt.IsZero()
		norm, err := Normalize(in, NormalizeOpts{MaxPayloadKB: s.maxPayloadKB, Now: s.nowFn()})
		if err != nil {
			return nil, err
		}
		// Store and compare the same value for an absent payload.
		if norm.Payload == nil {
			norm.Payload = []byte("{}")
		}
		ps[i] = Prepared{in: norm, createdAtDefaulted: createdAtDefaulted}
		if s.emb == nil || utf8.RuneCountInString(norm.EmbedText) < s.minEmbedLen {
			continue
		}
		if cached, ok := s.cache.GetDocument(norm.EmbedText); ok && len(cached) == s.emb.Dim() {
			ps[i].vec = cached
			continue
		}
		missIdx = append(missIdx, i)
		missTexts = append(missTexts, norm.EmbedText)
	}
	if len(missTexts) == 0 {
		return ps, nil
	}
	vecs, err := s.embed(ctx, missTexts)
	if err != nil {
		s.lg.Warn("embed failed; storing FTS-only", "err", err)
		return ps, nil
	}
	for j, i := range missIdx {
		s.cache.PutDocument(missTexts[j], vecs[j])
		ps[i].vec = vecs[j]
	}
	return ps, nil
}

// Commit persists one prepared item inside tx and returns its ID. Callers
// serialize writers; Commit runs the same-content short-circuit and upsert.
func (s *Service) Commit(ctx context.Context, tx *sql.Tx, p Prepared) (string, error) {
	norm := p.in
	existing, err := s.fetchExisting(ctx, tx, norm.ID)
	if err != nil {
		return "", err
	}
	// Resolve an unspecified tier: keep a live item's, else start hot.
	tier := DefaultTier
	if norm.Tier != nil {
		tier = *norm.Tier
	} else if existing != nil && !existing.Deleted {
		tier = existing.Tier
	}
	norm.Tier = &tier
	// A defaulted timestamp must not overwrite the original one.
	if p.createdAtDefaulted && existing != nil && !existing.Deleted {
		norm.Provenance.CreatedAt = existing.Provenance.CreatedAt
	}
	if existing != nil && !existing.Deleted && sameContent(*existing, norm) {
		// Unchanged content is a no-op, except that a vector missing from
		// an earlier failed embed is backfilled now that one is available.
		// Only the vec row is written: updated_at, created_at, access
		// stats, FTS and tags are untouched.
		if existing.HasVec || p.vec == nil {
			return norm.ID, nil
		}
		if _, err := tx.ExecContext(ctx,
			"INSERT OR REPLACE INTO vec(item_id, embedding) VALUES(?,?)",
			norm.ID, embed.EncodeVec(p.vec)); err != nil {
			return "", err
		}
		return norm.ID, nil
	}
	if err := s.upsert(ctx, tx, norm, p.vec, existing == nil || existing.Deleted); err != nil {
		return "", err
	}
	return norm.ID, nil
}

// CommitAll commits the prepared items in one transaction: either every
// item is written or none is. It returns the IDs in input order.
func (s *Service) CommitAll(ctx context.Context, ps []Prepared) ([]string, error) {
	tx, err := s.d.Raw.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	ids := make([]string, len(ps))
	for i, p := range ps {
		if ids[i], err = s.Commit(ctx, tx, p); err != nil {
			return nil, err
		}
	}
	return ids, tx.Commit()
}

// Remember is Prepare followed by CommitAll for a single input.
func (s *Service) Remember(ctx context.Context, in Input) (string, error) {
	ps, err := s.Prepare(ctx, []Input{in})
	if err != nil {
		return "", err
	}
	ids, err := s.CommitAll(ctx, ps)
	if err != nil {
		return "", err
	}
	return ids[0], nil
}

type storedItem struct {
	Kind       string
	Title      string
	Body       string
	Payload    []byte
	Provenance model.ProvDoc
	Tier       int
	Tags       []string
	Deleted    bool
	HasVec     bool // a row exists in vec for this item
}

// fetchExisting loads the stored item (live or tombstoned) for comparison
// with an incoming one; nil if the ID is unused.
func (s *Service) fetchExisting(ctx context.Context, q db.Querier, id string) (*storedItem, error) {
	var si storedItem
	var provJSON []byte
	var deletedAt sql.NullString
	err := q.QueryRowContext(ctx,
		`SELECT kind, title, body, payload, provenance, tier, deleted_at,
		        EXISTS(SELECT 1 FROM vec WHERE vec.item_id = items.id)
		 FROM items WHERE id=?`, id).
		Scan(&si.Kind, &si.Title, &si.Body, &si.Payload, &provJSON, &si.Tier, &deletedAt, &si.HasVec)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	si.Deleted = deletedAt.Valid
	si.Provenance = model.ParseProv(provJSON)
	if si.Tags, err = db.Tags(ctx, q, id); err != nil {
		return nil, err
	}
	return &si, nil
}

// sameContent reports whether re-remembering norm would change nothing
// stored. Stored tags come back sorted, hence the sorted comparison.
func sameContent(old storedItem, norm Input) bool {
	return old.Kind == norm.Kind && old.Title == norm.Title && old.Body == norm.Body &&
		bytes.Equal(old.Payload, norm.Payload) && old.Tier == *norm.Tier &&
		old.Provenance.Equal(norm.Provenance) &&
		slices.Equal(old.Tags, slices.Sorted(slices.Values(norm.Tags))) &&
		norm.Supersedes == "" // a supersession link is written by the upsert; never skip it
}

// embed embeds texts and checks the result shape.
func (s *Service) embed(ctx context.Context, texts []string) ([][]float32, error) {
	vecs, err := s.emb.Embed(ctx, texts)
	if err != nil {
		return nil, err
	}
	if len(vecs) != len(texts) {
		return nil, errs.ErrEmbedFailed
	}
	for _, v := range vecs {
		if len(v) != s.emb.Dim() {
			return nil, errs.ErrEmbedFailed
		}
	}
	return vecs, nil
}

// upsert writes the item row and replaces its vector, FTS row and tags.
// isNew is whether the item was absent or tombstoned before this call (not
// the RowsAffected of the upsert, which SQLite reports as 1 for both insert
// and conflict-update): a fresh or revived item starts with no access
// history and a new created_at, while an update keeps its own.
func (s *Service) upsert(ctx context.Context, tx *sql.Tx, norm Input, vec []float32, isNew bool) error {
	now := model.Stamp(s.nowFn())
	prov, _ := json.Marshal(norm.Provenance)
	// The item's existing FTS rowid, read first so a brand-new row (no FTS
	// row to replace) is told apart from a legacy row whose fts_rowid is NULL.
	var oldFTS sql.NullInt64
	rowExists := true
	if err := tx.QueryRowContext(ctx,
		"SELECT fts_rowid FROM items WHERE id=?", norm.ID).Scan(&oldFTS); errors.Is(err, sql.ErrNoRows) {
		rowExists = false
	} else if err != nil {
		return err
	}

	q := db.NewSeq(ctx, tx)
	q.Exec(`INSERT INTO items(id, kind, title, body, payload, provenance, tier, created_at, updated_at, access_count)
		 VALUES(?,?,?,?,?,?,?,?,?,0)
		 ON CONFLICT(id) DO UPDATE SET
		   kind=excluded.kind, title=excluded.title, body=excluded.body,
		   payload=excluded.payload, provenance=excluded.provenance,
		   tier=excluded.tier, updated_at=excluded.updated_at, deleted_at=NULL`,
		norm.ID, norm.Kind, norm.Title, norm.Body, norm.Payload, prov, *norm.Tier, now, now)

	q.Exec("DELETE FROM vec WHERE item_id=?", norm.ID)
	if vec != nil {
		q.Exec("INSERT INTO vec(item_id, embedding) VALUES(?,?)", norm.ID, embed.EncodeVec(vec))
	}

	// fts.item_id is UNINDEXED, so deleting by it scans the whole table:
	// delete by the recorded rowid instead (item_id scan only for legacy
	// rows without one) and reuse that rowid for the replacement. A nil
	// rowid lets FTS5 assign one, recorded on the item.
	var rowid any
	switch {
	case oldFTS.Valid:
		rowid = oldFTS.Int64
		q.Exec("DELETE FROM fts WHERE rowid=?", rowid)
	case rowExists:
		q.Exec("DELETE FROM fts WHERE item_id=?", norm.ID)
	}
	ftsRowid := q.Insert("INSERT INTO fts(rowid, item_id, title, body) VALUES(?,?,?,?)",
		rowid, norm.ID, norm.Title, norm.Body)
	if !oldFTS.Valid {
		q.Exec("UPDATE items SET fts_rowid=? WHERE id=?", ftsRowid, norm.ID)
	}

	q.Exec("DELETE FROM tags WHERE item_id=?", norm.ID)
	for _, t := range norm.Tags {
		q.Exec("INSERT OR IGNORE INTO tags(item_id, tag) VALUES(?,?)", norm.ID, t)
	}

	if norm.Supersedes != "" {
		if err := q.Err(); err != nil {
			return err
		}
		var live bool
		if err := tx.QueryRowContext(ctx,
			"SELECT EXISTS(SELECT 1 FROM items WHERE id=? AND deleted_at IS NULL)", norm.Supersedes).Scan(&live); err != nil {
			return err
		}
		if !live {
			return errs.ErrNotFound
		}
		q.Exec("UPDATE items SET superseded_by=?, updated_at=? WHERE id=?", norm.ID, now, norm.Supersedes)
	}
	if isNew {
		q.Exec("UPDATE items SET last_access=NULL, access_count=0, created_at=? WHERE id=?", now, norm.ID)
	}
	return q.Err()
}
