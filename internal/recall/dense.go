// Package recall orchestrates hybrid retrieval: a dense leg (flat
// brute-force KNN over stored vectors) and a sparse leg (FTS5 BM25),
// fused with a salience modulation into a normalized composite
// score, with graceful per-leg degradation, filtering, near-duplicate
// suppression, and token-budget shaping.
package recall

import (
	"context"
	"database/sql"
	"encoding/binary"
	"math"
	"sort"

	"github.com/alpdeniz/gomem/internal/db"
)

// hit is a single leg result. Score is a cosine similarity on the dense
// leg (higher is better) and an FTS5 BM25 rank on the sparse leg (lower is
// better).
type hit struct {
	ID    string
	Score float64
}

// denseKNN performs a flat brute-force nearest-neighbour search over
// every stored vector of items passing the filter (where, over alias "i"),
// returning the top-k by cosine similarity (ties broken by ID).
func denseKNN(ctx context.Context, d *db.DB, qvec []float32, topK int, where string, args []any) ([]hit, error) {
	rows, err := d.Raw.QueryContext(ctx,
		"SELECT v.item_id, v.embedding FROM vec v JOIN items i ON i.id = v.item_id WHERE "+where, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	qnorm := vecNorm(qvec)

	best := make([]hit, 0, topK)
	for rows.Next() {
		var id string
		var blob sql.RawBytes // valid until the next rows.Next()
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, err
		}
		if len(blob) != 4*len(qvec) {
			continue
		}
		sim := cosineBlob(qvec, qnorm, blob)
		best = pushBest(best, topK, hit{id, sim})
	}
	return best, rows.Err()
}

// vecNorm is the Euclidean norm of v.
func vecNorm(v []float32) float64 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	return math.Sqrt(n)
}

// cosineBlob is the cosine similarity of qvec (with norm qnorm) and a
// stored little-endian float32 vector of the same length; 0 if either is
// the zero vector.
func cosineBlob(qvec []float32, qnorm float64, blob []byte) float64 {
	var dot, vnorm float64
	for i, q := range qvec {
		x := float64(math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:])))
		dot += float64(q) * x
		vnorm += x * x
	}
	if qnorm == 0 || vnorm == 0 {
		return 0
	}
	return dot / (qnorm * math.Sqrt(vnorm))
}

// denseLookupChunk bounds the IN-list of one denseLookup query.
const denseLookupChunk = 500

// denseLookup returns the cosine similarity to qvec of each listed item's
// stored vector, by primary-key lookup in vec. Items without a vector, or
// whose vector has another dimension, are omitted.
func denseLookup(ctx context.Context, d *db.DB, qvec []float32, ids []string) (map[string]float64, error) {
	out := make(map[string]float64, len(ids))
	qnorm := vecNorm(qvec)
	for len(ids) > 0 {
		chunk := ids[:min(len(ids), denseLookupChunk)]
		ids = ids[len(chunk):]
		rows, err := d.Raw.QueryContext(ctx,
			"SELECT item_id, embedding FROM vec WHERE item_id IN ("+placeholders(len(chunk))+")", toArgs(chunk)...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var id string
			var blob sql.RawBytes // valid until the next rows.Next()
			if err := rows.Scan(&id, &blob); err != nil {
				rows.Close()
				return nil, err
			}
			if len(blob) == 4*len(qvec) {
				out[id] = cosineBlob(qvec, qnorm, blob)
			}
		}
		err = rows.Err()
		rows.Close()
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func better(a, b hit) bool { return a.Score > b.Score || (a.Score == b.Score && a.ID < b.ID) }

// pushBest inserts h into best, which is kept sorted (best hit first) and
// never exceeds topK entries.
func pushBest(best []hit, topK int, h hit) []hit {
	if len(best) == topK && !better(h, best[topK-1]) {
		return best
	}
	pos := sort.Search(len(best), func(i int) bool { return better(h, best[i]) })
	if len(best) < topK {
		best = append(best, hit{})
	}
	copy(best[pos+1:], best[pos:])
	best[pos] = h
	return best
}
