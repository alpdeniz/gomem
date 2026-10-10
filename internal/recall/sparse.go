package recall

import (
	"context"
	"strings"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/textutil"
)

const (
	// sparseFetchFactor and sparseFetchMin size the first FTS fetch relative
	// to the wanted hits; most filters keep most rows, so one round is
	// usually enough.
	sparseFetchFactor = 4
	sparseFetchMin    = 64
	// sparseGrowth multiplies the number of ranked rows fetched so far on
	// each further round, so a very selective filter needs only a few
	// re-rankings.
	sparseGrowth = 8
	// sparseChunkMin and sparseChunkMax bound how many candidate IDs one
	// filter lookup checks.
	sparseChunkMin = 32
	sparseChunkMax = 500
)

// sparseMatch runs an FTS5 MATCH query, returning up to topK rows ordered
// by relevance (ascending BM25, where lower is better), restricted to items
// passing the filter (where, over alias "i").
//
// FTS5 ranks the matches alone (no join, which at 10^4+ matching rows costs
// more than the ranking itself) and the best ones are then filtered in
// chunks, so the filter runs on roughly topK rows instead of every match.
// If too few pass, further, larger slices of the ranking are fetched until
// topK survive or the matches run out, so a selective filter still finds
// rows ranked far down.
func sparseMatch(ctx context.Context, d *db.DB, query string, topK int, where string, args []any) ([]hit, error) {
	if query == "" || topK <= 0 {
		return nil, nil
	}
	var out []hit
	seen := map[string]bool{}
	offset, limit := 0, max(topK*sparseFetchFactor, sparseFetchMin)
	for {
		ranked, err := ftsRanked(ctx, d, query, limit, offset)
		if err != nil {
			return nil, err
		}
		for i := 0; i < len(ranked) && len(out) < topK; {
			end := min(i+min(max(2*(topK-len(out)), sparseChunkMin), sparseChunkMax), len(ranked))
			live, err := passing(ctx, d, hitIDs(ranked[i:end]), where, args)
			if err != nil {
				return nil, err
			}
			for _, h := range ranked[i:end] {
				if live[h.ID] && !seen[h.ID] && len(out) < topK {
					seen[h.ID] = true
					out = append(out, h)
				}
			}
			i = end
		}
		if len(out) >= topK || len(ranked) < limit {
			return out, nil
		}
		offset += limit
		limit = offset * sparseGrowth
	}
}

// ftsRanked returns FTS5 matches ranked by ascending BM25, with no
// filtering. The rowid tiebreak keeps pages consistent across offsets.
func ftsRanked(ctx context.Context, d *db.DB, query string, limit, offset int) ([]hit, error) {
	// fts is not aliased: bm25() must be given the table name.
	rows, err := d.Raw.QueryContext(ctx,
		"SELECT item_id, bm25(fts) AS score FROM fts WHERE fts MATCH ? ORDER BY score ASC, rowid ASC LIMIT ? OFFSET ?",
		query, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]hit, 0, min(limit, 1024))
	for rows.Next() {
		var h hit
		if err := rows.Scan(&h.ID, &h.Score); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

// buildFTSQuery turns free text into a safe FTS5 query expression by
// extracting letter/digit tokens and joining them with OR. OR gives the
// sparse leg its role as a breadth-oriented recall channel: it surfaces
// documents matching any query term, while BM25 ranking still orders results
// by relevance (documents matching more terms rank higher). The dense leg
// provides the complementary precision (semantic similarity).
//
// Common English stopwords are dropped: in natural-language goals they match
// nearly every row, which makes the query cost scale with the corpus while
// adding no ranking signal. A query made only of stopwords keeps them.
func buildFTSQuery(text string) string {
	tokens := textutil.ContentWords(text)
	if len(tokens) == 0 {
		return ""
	}
	quoted := make([]string, 0, len(tokens))
	for _, t := range tokens {
		quoted = append(quoted, "\""+strings.ReplaceAll(t, "\"", "\"\"")+"\"")
	}
	return strings.Join(quoted, " OR ")
}

func hitIDs(hits []hit) []string {
	ids := make([]string, len(hits))
	for i, h := range hits {
		ids[i] = h.ID
	}
	return ids
}
