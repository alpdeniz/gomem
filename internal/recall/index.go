package recall

import (
	"context"
	"database/sql"
	"encoding/binary"
	"math"
	"slices"
	"sync"

	"github.com/alpdeniz/gomem/internal/db"
)

// vecIndex is an in-memory copy of the live rows of the vec table: one
// row-major matrix of L2-normalized vectors, so cosine similarity is a
// plain dot product. Cost is N*dim*4 bytes.
//
// It is loaded lazily and kept in step with the facade's own writes (see
// Engine.IndexWrite). Writes by anyone else are caught with PRAGMA
// data_version, which changes whenever another connection commits; since
// the value is per connection, one connection is pinned for the index's
// lifetime and any change drops the index for a lazy reload.
//
// The index holds rows of exactly one dimension, dim: the length of the
// latest query vector (the embedder's real width, which may differ from the
// configured one until its first embed), initially a guess from the
// embedder. Stored rows of any other width are left out, as in the SQLite
// path. A query of another width reloads the index at that width.
type vecIndex struct {
	d *db.DB

	mu      sync.RWMutex // guards everything below
	dim     int          // width of every row in mat
	loaded  bool
	ids     []string
	mat     []float32
	pos     map[string]int
	conn    *sql.Conn // pinned; non-nil whenever loaded
	version int64     // conn's data_version when the index was last in step
}

// rankMin is the least number of candidates ranked per pass.
const rankMin = 256

// dataVersion reads the pinned connection's PRAGMA data_version.
func (x *vecIndex) dataVersion(ctx context.Context) (v int64, err error) {
	err = x.conn.QueryRowContext(ctx, "PRAGMA data_version").Scan(&v)
	return
}

// fresh reports whether the index is loaded and no other connection has
// committed since.
func (x *vecIndex) fresh(ctx context.Context) bool {
	if !x.loaded {
		return false
	}
	v, err := x.dataVersion(ctx)
	return err == nil && v == x.version
}

func (x *vecIndex) drop() {
	x.loaded = false
	x.ids, x.mat, x.pos = nil, nil, nil
}

// load replaces the index with the live vectors in the database. The
// version is read first, so a commit racing the scan just forces one more
// reload.
func (x *vecIndex) load(ctx context.Context) error {
	x.drop()
	if x.conn == nil {
		c, err := x.d.Raw.Conn(ctx)
		if err != nil {
			return err
		}
		x.conn = c
	}
	v, err := x.dataVersion(ctx)
	if err != nil {
		return err
	}
	rows, err := x.d.Raw.QueryContext(ctx,
		"SELECT v.item_id, v.embedding FROM vec v JOIN items i ON i.id = v.item_id WHERE i.deleted_at IS NULL")
	if err != nil {
		return err
	}
	defer rows.Close()
	x.pos = map[string]int{}
	for rows.Next() {
		var id string
		var blob sql.RawBytes // valid until the next rows.Next()
		if err := rows.Scan(&id, &blob); err != nil {
			x.drop()
			return err
		}
		x.set(id, blob)
	}
	if err := rows.Err(); err != nil {
		x.drop()
		return err
	}
	x.version, x.loaded = v, true
	return nil
}

// set inserts or replaces id's vector from its stored encoding; a blob of
// the wrong length (another dimension) removes it, as the SQLite path skips
// such rows.
func (x *vecIndex) set(id string, blob []byte) {
	if x.dim <= 0 || len(blob) != 4*x.dim {
		x.remove(id)
		return
	}
	i, ok := x.pos[id]
	if !ok {
		i = len(x.ids)
		x.ids = append(x.ids, id)
		x.mat = slices.Grow(x.mat, x.dim)[:len(x.mat)+x.dim]
		x.pos[id] = i
	}
	row := x.mat[i*x.dim : (i+1)*x.dim]
	for j := range row {
		row[j] = math.Float32frombits(binary.LittleEndian.Uint32(blob[j*4:]))
	}
	unitize(row)
}

// unitize scales v in place to unit length (a zero vector stays zero).
func unitize(v []float32) {
	var norm float64
	for _, f := range v {
		norm += float64(f) * float64(f)
	}
	if norm > 0 {
		s := float32(1 / math.Sqrt(norm))
		for i := range v {
			v[i] *= s
		}
	}
}

// remove deletes id by moving the last row into its slot.
func (x *vecIndex) remove(id string) {
	i, ok := x.pos[id]
	if !ok {
		return
	}
	last := len(x.ids) - 1
	if i != last {
		copy(x.mat[i*x.dim:(i+1)*x.dim], x.mat[last*x.dim:])
		x.ids[i] = x.ids[last]
		x.pos[x.ids[i]] = i
	}
	x.ids = x.ids[:last]
	x.mat = x.mat[:last*x.dim]
	delete(x.pos, id)
}

// search returns the top-k hits by cosine similarity among items passing
// the filter (where, over alias "i"), reloading the index first if it is
// stale or was built for another vector width.
func (x *vecIndex) search(ctx context.Context, qvec []float32, k int, where string, args []any) ([]hit, error) {
	if len(qvec) == 0 {
		return nil, nil
	}
	x.mu.RLock()
	if x.dim == len(qvec) && x.fresh(ctx) {
		defer x.mu.RUnlock()
		return x.rank(ctx, qvec, k, where, args)
	}
	x.mu.RUnlock()
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.dim != len(qvec) {
		x.drop()
		x.dim = len(qvec)
	}
	if !x.fresh(ctx) {
		if err := x.load(ctx); err != nil {
			return nil, err
		}
	}
	return x.rank(ctx, qvec, k, where, args)
}

// rank scores every row, then walks the best n in chunks, keeping those
// that pass the filter until k survive; n grows when the filter is
// selective, and the walk ends once every row has been ranked. A chunk is
// only as long as the survivors still missing (at least 32), since each
// candidate costs an indexed lookup.
func (x *vecIndex) rank(ctx context.Context, qvec []float32, k int, where string, args []any) ([]hit, error) {
	if x.dim <= 0 || len(qvec) != x.dim {
		return nil, nil // not this index's width; no row can match
	}
	rows := min(len(x.ids), len(x.mat)/x.dim)
	q := slices.Clone(qvec)
	unitize(q)
	var out []hit
	checked := 0
	for n := max(4*k, rankMin); ; n *= 4 {
		top := make([]hit, 0, min(n, rows))
		for i, id := range x.ids[:rows] {
			sim := float64(dot32(q, x.mat[i*x.dim:(i+1)*x.dim]))
			top = pushBest(top, n, hit{id, sim})
		}
		for checked < len(top) && len(out) < k {
			end := min(checked+max(k-len(out), 32), len(top))
			live, err := passing(ctx, x.d, hitIDs(top[checked:end]), where, args)
			if err != nil {
				return nil, err
			}
			for _, h := range top[checked:end] {
				if live[h.ID] && len(out) < k {
					out = append(out, h)
				}
			}
			checked = end
		}
		if len(out) >= k || len(top) < n {
			return out, nil
		}
	}
}

// dot32 is the dot product over the common prefix of a and b, so it can
// neither read past either slice nor panic on a length mismatch.
func dot32(a, b []float32) float32 {
	n := min(len(a), len(b))
	a, b = a[:n], b[:n]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// IndexWrite runs a write that commits through the database and returns
// the IDs whose vector or liveness may have changed, then brings the
// in-memory index in step with them (it reads the rows back, so the index
// always mirrors the database). The caller must serialize writers. With the
// index off it just runs write.
//
// Own commits move data_version too, so before the write any earlier
// foreign commit is detected and drops the index, and after it the new
// version is adopted. A foreign commit landing between our commit and that
// read would go unnoticed until the next write or Reflect; the window is
// microseconds.
func (e *Engine) IndexWrite(ctx context.Context, write func() ([]string, error)) ([]string, error) {
	x := e.idx
	if x == nil {
		return write()
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.loaded && !x.fresh(ctx) {
		x.drop()
	}
	ids, err := write()
	if err != nil || !x.loaded {
		return ids, err
	}
	for _, id := range ids {
		var blob []byte
		err := x.d.Raw.QueryRowContext(ctx,
			"SELECT v.embedding FROM vec v JOIN items i ON i.id = v.item_id WHERE v.item_id = ? AND i.deleted_at IS NULL", id).Scan(&blob)
		switch {
		case err == sql.ErrNoRows:
			x.remove(id)
		case err != nil:
			x.drop()
			return ids, nil
		default:
			x.set(id, blob)
		}
	}
	if v, err := x.dataVersion(ctx); err != nil {
		x.drop()
	} else {
		x.version = v
	}
	return ids, nil
}

// InvalidateIndex drops the in-memory index; the next query reloads it.
func (e *Engine) InvalidateIndex() {
	if e.idx == nil {
		return
	}
	e.idx.mu.Lock()
	e.idx.drop()
	e.idx.mu.Unlock()
}

// Close releases the index's pinned connection.
func (e *Engine) Close() {
	if e.idx == nil {
		return
	}
	e.idx.mu.Lock()
	defer e.idx.mu.Unlock()
	e.idx.drop()
	if e.idx.conn != nil {
		e.idx.conn.Close()
		e.idx.conn = nil
	}
}
