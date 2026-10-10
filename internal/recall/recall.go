package recall

import (
	"context"
	"database/sql"
	"log/slog"
	"math"
	"sort"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/alpdeniz/gomem/internal/db"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/model"
	"github.com/alpdeniz/gomem/internal/textutil"
)

// Query is the internal recall request.
type Query struct {
	Goal              string
	SubGoal           string
	Kinds             []string
	MinTier           int
	Tags              []string
	MinConfidence     float64
	TopK              int
	Budget            int
	HalfLifeDays      float64
	IncludeSuperseded bool
	DedupJaccard      float64

	// Set by Prepare: the goal's embedding (nil: dense leg skipped).
	qvec     []float32
	prepared bool
}

// Candidate is a retrieved item carrying its leg scores and the
// metadata needed for scoring and shaping.
type Candidate struct {
	ID         string
	Kind       string
	Title      string
	Body       string
	Provenance model.ProvDoc
	Tier       int
	model.Usage
	Superseded  bool
	DenseScore  float64
	HasDense    bool
	SparseScore float64
	HasSparse   bool
	Salience    float64
	Score       float64
}

// Result is a scored, shaped recall output.
type Result struct {
	ID         string
	Kind       string
	Title      string
	Body       string
	Score      float64
	Provenance model.ProvDoc
	Tier       int
	Tokens     int
}

// EngineConfig tunes the recall engine.
type EngineConfig struct {
	Dim           int
	AlphaDense    float64
	BetaSparse    float64
	GammaSalience float64
	model.Params  // the salience leg; a query may override HalfLifeDays
	DefaultTopK   int
	DenseIndex    string // "memory" keeps vectors in RAM; anything else scans SQLite
	FusionMode    string // how leg scores are normalized: "legacy" (default), "minmax", "rrf"
}

// Engine runs hybrid recall.
type Engine struct {
	d     *db.DB
	emb   embed.Embedder
	cache *embed.Cache
	lg    *slog.Logger
	cfg   EngineConfig
	idx   *vecIndex // nil unless DenseIndex is "memory"
	nowFn func() time.Time
}

// NewEngine wires a recall engine. The cache memoizes goal embeddings;
// nil gives a private in-memory cache.
func NewEngine(d *db.DB, emb embed.Embedder, cache *embed.Cache, lg *slog.Logger, cfg EngineConfig) *Engine {
	if lg == nil {
		lg = slog.New(slog.DiscardHandler)
	}
	if cache == nil {
		cache = embed.NewCache("", "")
	}
	e := &Engine{d: d, emb: emb, cache: cache, lg: lg, cfg: cfg, nowFn: time.Now}
	if cfg.DenseIndex == "memory" && emb != nil {
		e.idx = &vecIndex{d: d, dim: emb.Dim()} // a guess; search re-aligns it to the query width
	}
	return e
}

// embedGoal returns the goal's embedding, consulting the cache first. A
// nil vector with a nil error means the embedder returned an unusable
// shape, so the dense leg is skipped.
func (e *Engine) embedGoal(ctx context.Context, goal string) ([]float32, error) {
	if v, ok := e.cache.Get(goal); ok && len(v) == e.emb.Dim() {
		return v, nil
	}
	vecs, err := e.emb.Embed(ctx, []string{goal})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 || len(vecs[0]) != e.emb.Dim() {
		return nil, nil
	}
	e.cache.Put(goal, vecs[0])
	return vecs[0], nil
}

// Prepare embeds the query goal, the one step of Run that needs no
// database, so a caller can do it before taking a lock that guards the
// database. An embedding failure is logged and leaves the query
// keyword-only, exactly as Run would. Run embeds for itself when handed an
// unprepared query.
func (e *Engine) Prepare(ctx context.Context, q Query) Query {
	q.prepared, q.qvec = true, nil
	if e.emb == nil || q.Goal == "" {
		return q
	}
	qvec, err := e.embedGoal(ctx, q.Goal)
	if err != nil {
		e.lg.Warn("query embed failed; dense leg skipped", "err", err)
		return q
	}
	q.qvec = qvec
	return q
}

// Run executes hybrid recall, returning shaped results sorted by
// descending score. A failing leg degrades gracefully (skipped) rather
// than failing the whole call, unless ctx is done, which is an error.
func (e *Engine) Run(ctx context.Context, q Query) ([]Result, error) {
	if !q.prepared {
		q = e.Prepare(ctx, q)
	}
	if q.TopK <= 0 {
		q.TopK = e.cfg.DefaultTopK
	}
	q.Tags = textutil.CleanTags(q.Tags)
	cands, err := e.retrieve(ctx, q)
	if err != nil || len(cands) == 0 {
		return nil, err
	}
	e.score(ctx, cands, q.HalfLifeDays)
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Score != cands[j].Score {
			return cands[i].Score > cands[j].Score
		}
		return cands[i].ID < cands[j].ID
	})
	return shape(dedupNear(cands, q.DedupJaccard), q.TopK, q.Budget), nil
}

// retrieve runs both legs, loads the union of their hits, and attaches each
// leg's score normalized to [0,1] per the fusion mode.
func (e *Engine) retrieve(ctx context.Context, q Query) ([]Candidate, error) {
	fetchLimit := max(q.TopK*3, 10)
	where, whereArgs := filterSQL(q, "i")

	// The legs are independent, so they run concurrently: latency is the
	// slower leg's, not the sum. Each writes only its own map.
	var dense, sparse map[string]float64
	runConcurrently(
		func() { dense = e.denseLeg(ctx, q.qvec, fetchLimit, where, whereArgs) },
		func() { sparse = e.sparseLeg(ctx, q, fetchLimit, where, whereArgs) },
	)
	if err := ctx.Err(); err != nil {
		return nil, err // a cancelled leg is not an empty one
	}
	if e.legacyFusion() && q.qvec != nil {
		e.scoreSparseOnly(ctx, q.qvec, dense, sparse)
	}

	ids := make(map[string]bool, len(dense)+len(sparse))
	for id := range dense {
		ids[id] = true
	}
	for id := range sparse {
		ids[id] = true
	}
	if len(ids) == 0 {
		return nil, nil
	}
	cands, err := e.loadCandidates(ctx, ids, q)
	if err != nil {
		return nil, err
	}
	denseNorm, sparseNorm := e.normalizeLegs(dense, sparse)
	for i := range cands {
		c := &cands[i]
		c.DenseScore, c.HasDense = denseNorm[c.ID]
		c.SparseScore, c.HasSparse = sparseNorm[c.ID]
	}
	return cands, nil
}

// score fuses the legs with the salience modulation into each candidate's
// [0,1] composite. halfLifeDays > 0 overrides the configured recency
// half-life for this query.
func (e *Engine) score(ctx context.Context, cands []Candidate, halfLifeDays float64) {
	params := e.cfg.Params
	if halfLifeDays > 0 {
		params.HalfLifeDays = halfLifeDays
	}
	now := e.nowFn()
	weightSum := e.cfg.AlphaDense + e.cfg.BetaSparse + e.cfg.GammaSalience
	for i := range cands {
		c := &cands[i]
		c.Salience = params.Of(c.Usage, now)
		raw := e.cfg.AlphaDense*c.DenseScore + e.cfg.BetaSparse*c.SparseScore + e.cfg.GammaSalience*c.Salience
		c.Score = raw / weightSum
		if c.Superseded {
			c.Score *= 0.6
		}
		c.Score = min(max(c.Score, 0), 1)
	}
	if e.lg.Enabled(ctx, slog.LevelDebug) {
		e.logLegContributions(cands)
	}
}

// shape trims the ranked candidates to topK and then to the token budget
// (the first result is always kept).
func shape(cands []Candidate, topK, budget int) []Result {
	cands = cands[:min(len(cands), topK)]
	out := make([]Result, 0, len(cands))
	total := 0
	for _, c := range cands {
		toks := estimateTokens(c.Title + " " + c.Body)
		if budget > 0 && len(out) > 0 && total+toks > budget {
			break
		}
		total += toks
		out = append(out, Result{
			ID: c.ID, Kind: c.Kind, Title: c.Title, Body: c.Body,
			Score: c.Score, Provenance: c.Provenance, Tier: c.Tier, Tokens: toks,
		})
	}
	return out
}

// runConcurrently runs a and b in parallel and returns when both are done,
// so neither outlives the call. A panic in either is re-raised on the
// caller's goroutine, where the caller's recovery (if any) can see it.
func runConcurrently(a, b func()) {
	var wg sync.WaitGroup
	var bPanic any
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() { bPanic = recover() }()
		b()
	}()
	func() {
		defer wg.Wait()
		a()
	}()
	if bPanic != nil {
		panic(bPanic)
	}
}

// denseLeg returns raw cosine similarities of the top fetchLimit items, or
// none when there is no query vector or the leg fails.
func (e *Engine) denseLeg(ctx context.Context, qvec []float32, fetchLimit int, where string, args []any) map[string]float64 {
	if qvec == nil {
		return map[string]float64{}
	}
	var hits []hit
	var err error
	if e.idx != nil {
		hits, err = e.idx.search(ctx, qvec, fetchLimit, where, args)
	} else {
		hits, err = denseKNN(ctx, e.d, qvec, fetchLimit, where, args)
	}
	return e.legScores("dense", hits, err)
}

// sparseLeg returns raw FTS5 BM25 scores (lower is better), or none when
// the query has no usable terms or the leg fails.
func (e *Engine) sparseLeg(ctx context.Context, q Query, fetchLimit int, where string, args []any) map[string]float64 {
	// SubGoal only widens the sparse query; the dense leg stays anchored on Goal.
	ftsQuery := buildFTSQuery(q.Goal + " " + q.SubGoal)
	if ftsQuery == "" {
		return map[string]float64{}
	}
	hits, err := sparseMatch(ctx, e.d, ftsQuery, fetchLimit, where, args)
	return e.legScores("sparse", hits, err)
}

// legScores maps a leg's hits to scores by ID; a failed leg is logged and
// contributes none.
func (e *Engine) legScores(leg string, hits []hit, err error) map[string]float64 {
	scores := make(map[string]float64, len(hits))
	if err != nil {
		e.lg.Warn(leg+" leg failed", "err", err)
		return scores
	}
	for _, h := range hits {
		scores[h.ID] = h.Score
	}
	return scores
}

// legacyFusion reports whether leg scores use the default absolute
// normalization (dense (sim+1)/2) rather than a rank-based one.
func (e *Engine) legacyFusion() bool {
	return e.cfg.FusionMode != "minmax" && e.cfg.FusionMode != "rrf"
}

// scoreSparseOnly gives each item that only the sparse leg found its real
// cosine similarity to qvec, added to dense. Otherwise it would score 0
// where the lowest possible real similarity maps to 0 and typical ones to
// 0.5+, penalizing exact keyword matches (identifiers) that the dense leg
// ranked outside its top hits. Only for legacy fusion: minmax and rrf are
// relative to the dense hits' own range or rank, where "below every dense
// hit" is already what a sparse-only item is. Items without a comparable
// stored vector keep no dense score, and a failed lookup just skips this.
func (e *Engine) scoreSparseOnly(ctx context.Context, qvec []float32, dense, sparse map[string]float64) {
	var ids []string
	for id := range sparse {
		if _, ok := dense[id]; !ok {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return
	}
	sims, err := denseLookup(ctx, e.d, qvec, ids)
	if err != nil {
		e.lg.Warn("sparse-only dense lookup failed", "err", err)
		return
	}
	for id, sim := range sims {
		dense[id] = sim
	}
}

// loadCandidates fetches full item data for the candidate IDs, re-applying
// the query's filters so stale or unfiltered IDs can never leak through.
func (e *Engine) loadCandidates(ctx context.Context, ids map[string]bool, q Query) ([]Candidate, error) {
	idList := make([]string, 0, len(ids))
	for id := range ids {
		idList = append(idList, id)
	}
	filter, filterArgs := filterSQL(q, "items")
	query := "SELECT id, kind, title, body, provenance, tier, created_at, updated_at, last_access, access_count, " +
		supersededExpr("items") + " FROM items" + byPK +
		" WHERE id IN (" + placeholders(len(idList)) + ") AND " + filter
	rows, err := e.d.Raw.QueryContext(ctx, query, append(toArgs(idList), filterArgs...)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	now := e.nowFn()
	var out []Candidate
	for rows.Next() {
		var (
			c                    Candidate
			prov                 []byte
			createdAt, updatedAt string
			lastAccess           sql.NullString
			accessCount          int
		)
		if err := rows.Scan(&c.ID, &c.Kind, &c.Title, &c.Body, &prov, &c.Tier, &createdAt, &updatedAt, &lastAccess, &accessCount, &c.Superseded); err != nil {
			return nil, err
		}
		c.Provenance = model.ParseProv(prov)
		c.Usage = model.NewUsage(c.Provenance.Confidence, accessCount, createdAt, updatedAt, lastAccess, now)
		out = append(out, c)
	}
	return out, rows.Err()
}

// estimateTokens gives a cheap token-count approximation (~4 chars/token).
// It counts Unicode code points (runes) rather than bytes so that
// multi-byte characters (CJK, emoji) are not over-weighted.
func estimateTokens(s string) int {
	n := utf8.RuneCountInString(s) / 4
	if n == 0 && s != "" {
		n = 1
	}
	return n
}

// normalizeLegs maps each leg's raw scores (cosine similarity; FTS5 BM25,
// where more negative is more relevant) onto [0,1] per the fusion mode:
//
//	legacy: dense (sim+1)/2; sparse min-max scaled over its batch
//	minmax: both legs min-max scaled over their hits
//	rrf:    reciprocal rank per leg, 1/(60+rank), as a share of the best rank's
func (e *Engine) normalizeLegs(dense, sparse map[string]float64) (dn, sn map[string]float64) {
	switch e.cfg.FusionMode {
	case "minmax":
		return minMax(dense), normalizeSparse(sparse)
	case "rrf":
		return rrf(dense), rrf(negate(sparse))
	}
	dn = make(map[string]float64, len(dense))
	for id, sim := range dense {
		dn[id] = (sim + 1) / 2
	}
	return dn, normalizeSparse(sparse)
}

// normalizeSparse maps raw FTS5 BM25 scores onto [0,1] so the sparse leg is
// commensurate with the dense leg. FTS5 ranks by negative BM25 (more
// negative == more relevant), so we negate first (larger == better) and
// then min-max scale across the fetched batch.
func normalizeSparse(scores map[string]float64) map[string]float64 {
	return minMax(negate(scores))
}

func negate(scores map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(scores))
	for id, s := range scores {
		out[id] = -s
	}
	return out
}

// minMax scales scores (larger == better) onto [0,1]. A degenerate
// (all-tied) batch maps to a neutral 0.5 so it neither inflates nor
// collapses.
func minMax(scores map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(scores))
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, s := range scores {
		lo, hi = math.Min(lo, s), math.Max(hi, s)
	}
	for id, s := range scores {
		if hi <= lo {
			out[id] = 0.5
		} else {
			out[id] = (s - lo) / (hi - lo)
		}
	}
	return out
}

// rrf scores each id by 1/(60+rank) over its rank in scores (larger ==
// better, ties by ID), divided by the rank-1 value so the best maps to 1.
func rrf(scores map[string]float64) map[string]float64 {
	ids := make([]string, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if scores[ids[i]] != scores[ids[j]] {
			return scores[ids[i]] > scores[ids[j]]
		}
		return ids[i] < ids[j]
	})
	out := make(map[string]float64, len(ids))
	for r, id := range ids {
		out[id] = 61 / float64(61+r)
	}
	return out
}

// mean accumulates a running average.
type mean struct {
	sum float64
	n   int
}

func (m *mean) add(x float64) { m.sum += x; m.n++ }

func (m mean) avg() float64 {
	if m.n == 0 {
		return 0
	}
	return m.sum / float64(m.n)
}

// logLegContributions reports each leg's realized influence on the
// composite: its nominal weight multiplied by the leg's mean normalized
// score, expressed as a share of the total. The shares reveal the true
// contribution split (which drifts from the nominal alpha/beta/gamma as
// leg scores vary) so the weights can be calibrated against behaviour.
func (e *Engine) logLegContributions(cands []Candidate) {
	var dense, sparse, sal mean
	for _, c := range cands {
		if c.HasDense {
			dense.add(c.DenseScore)
		}
		if c.HasSparse {
			sparse.add(c.SparseScore)
		}
		sal.add(c.Salience)
	}
	shares := [3]float64{e.cfg.AlphaDense * dense.avg(), e.cfg.BetaSparse * sparse.avg(), e.cfg.GammaSalience * sal.avg()}
	total := shares[0] + shares[1] + shares[2]
	pct := func(share float64) float64 {
		if total <= 0 {
			return 0
		}
		return round3(100 * share / total)
	}
	e.lg.Debug("recall leg contributions",
		"dense_mean", round3(dense.avg()), "dense_share_pct", pct(shares[0]),
		"sparse_mean", round3(sparse.avg()), "sparse_share_pct", pct(shares[1]),
		"salience_mean", round3(sal.avg()), "salience_share_pct", pct(shares[2]),
		"dense_n", dense.n, "sparse_n", sparse.n, "salience_n", sal.n,
	)
}

// round3 rounds to the third decimal for compact log output.
func round3(x float64) float64 {
	return math.Round(x*1000) / 1000
}
