// Package main is a thin cgo bridge exposing the gomem API to
// dynamic languages (Python via ctypes). Build with:
//
//	sh python/build.sh   (CGO_ENABLED=1 go build -buildmode=c-shared, per-OS lib name)
//
// Convention: every entry point takes and returns NUL-terminated
// UTF-8 JSON strings. Success payloads carry no "error" key;
// failures return {"error": "<message>"}. Returned strings are
// heap-owned and must be released with GomemFreeString.
package main

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/alpdeniz/gomem"
)

// #include <stdlib.h>
import "C"

// ---------------------------------------------------------------------------
// Wire DTOs (snake_case JSON, mirroring the public Go types)
// ---------------------------------------------------------------------------

type provDoc struct {
	Source   string `json:"source"`
	RunID    string `json:"run_id,omitempty"`
	NodeRole string `json:"node_role,omitempty"`
	// Pointer: absent means unspecified (stored as 1); an explicit 0 is kept.
	Confidence   *float64 `json:"confidence,omitempty"`
	EvidenceRefs []string `json:"evidence_refs,omitempty"`
	CreatedAt    string   `json:"created_at,omitempty"` // RFC3339
}

type itemDoc struct {
	ID         string  `json:"id,omitempty"`
	Kind       string  `json:"kind"`
	Title      string  `json:"title"`
	Body       string  `json:"body"`
	Payload    any     `json:"payload,omitempty"`
	Provenance provDoc `json:"provenance"`
	// Tier is a pointer so an absent field (unspecified: hot for a new
	// item, the stored tier kept on update) is distinguishable from an
	// explicit 0 (cold). It maps straight onto gomem.Item.Tier.
	Tier         *int     `json:"tier,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	Supersedes   string   `json:"supersedes,omitempty"`
	SupersededBy string   `json:"superseded_by,omitempty"`
}

type queryDoc struct {
	Goal                  string   `json:"goal"`
	SubGoal               string   `json:"sub_goal,omitempty"`
	Kinds                 []string `json:"kinds,omitempty"`
	MinTier               int      `json:"min_tier"`
	Tags                  []string `json:"tags,omitempty"`
	MinConfidence         float64  `json:"min_confidence"`
	TopK                  int      `json:"top_k"`
	BudgetTokens          int      `json:"budget_tokens"`
	RecencyHalfLifeDays   float64  `json:"recency_half_life_days"`
	IncludeSuperseded     bool     `json:"include_superseded"`
	DedupJaccardThreshold float64  `json:"dedup_jaccard_threshold"`
}

type hitDoc struct {
	ID         string  `json:"id"`
	Kind       string  `json:"kind"`
	Title      string  `json:"title"`
	Body       string  `json:"body"`
	Score      float64 `json:"score"`
	Tier       int     `json:"tier"`
	Tokens     int     `json:"tokens"`
	Provenance provDoc `json:"provenance"`
}

// configDoc numeric fields are pointers so an explicit 0 (e.g.
// alpha_dense=0, max_retries=0) is distinguishable from "not set".
type configDoc struct {
	Path                  string   `json:"path"`
	Dim                   *int     `json:"dim"`
	Provider              string   `json:"provider"`
	APIURL                string   `json:"api_url"`
	APIKey                string   `json:"api_key"`
	Model                 string   `json:"model"`
	TimeoutMS             *int     `json:"timeout_ms"`
	MaxRetries            *int     `json:"max_retries"`
	BatchSize             *int     `json:"batch_size"`
	CacheDir              string   `json:"cache_dir"`
	DenseIndex            string   `json:"dense_index"`
	FusionMode            string   `json:"fusion_mode"`
	DefaultTopK           *int     `json:"top_k"`
	RecencyHalfLifeDays   *float64 `json:"half_life_days"`
	DedupJaccardThreshold *float64 `json:"dedup_jaccard"`
	AlphaDense            *float64 `json:"alpha_dense"`
	BetaSparse            *float64 `json:"beta_sparse"`
	GammaSalience         *float64 `json:"gamma_salience"`
	WConfidence           *float64 `json:"w_confidence"`
	WFrequency            *float64 `json:"w_frequency"`
	WRecency              *float64 `json:"w_recency"`
	FreqCap               *int     `json:"freq_cap"`
	EvictionAgeDays       *int     `json:"eviction_age_days"`
	MaxItems              *int     `json:"max_items"`
	MaxPayloadKB          *int     `json:"max_payload_kb"`
	EmbedLengthThreshold  *int     `json:"embed_length_threshold"`
	SendDimensions        *bool    `json:"send_dimensions"`
	LogLevel              string   `json:"log_level"`
}

// ---------------------------------------------------------------------------
// Handle registry
// ---------------------------------------------------------------------------

var (
	handlesMu  sync.Mutex
	handles            = map[uintptr]gomem.Memory{}
	nextHandle uintptr = 1
)

func register(m gomem.Memory) uintptr {
	handlesMu.Lock()
	defer handlesMu.Unlock()
	handles[nextHandle] = m
	id := nextHandle
	nextHandle++
	return id
}

func acquire(id uintptr) (gomem.Memory, bool) {
	handlesMu.Lock()
	defer handlesMu.Unlock()
	m, ok := handles[id]
	return m, ok
}

func deregister(id uintptr) {
	handlesMu.Lock()
	delete(handles, id)
	handlesMu.Unlock()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func goStr(p *C.char) string {
	if p == nil {
		return ""
	}
	return C.GoString(p)
}

// callTimeout bounds every bridge call.
const callTimeout = 5 * time.Minute

// run executes fn and marshals its outcome (result or error) into a
// freshly allocated C string the caller must free.
func run(fn func() (any, error)) *C.char {
	return C.CString(string(runJSON(fn)))
}

// runJSON executes fn and returns its JSON outcome. A panic is
// converted into an error document: unwinding across cgo would kill
// the host process.
func runJSON(fn func() (any, error)) (b []byte) {
	defer func() {
		if r := recover(); r != nil {
			b, _ = json.Marshal(map[string]string{"error": fmt.Sprintf("panic: %v", r)})
		}
	}()
	out, err := fn()
	if err != nil {
		out = map[string]any{"error": err.Error()}
	}
	b, jerr := json.Marshal(out)
	if jerr != nil {
		b, _ = json.Marshal(map[string]string{"error": jerr.Error()})
	}
	return b
}

// call runs fn against the Memory behind handle, bounded by callTimeout.
func call(handle C.longlong, fn func(ctx context.Context, m gomem.Memory) (any, error)) *C.char {
	return run(func() (any, error) {
		m, ok := acquire(uintptr(handle))
		if !ok {
			return nil, fmt.Errorf("unknown handle %d", int64(handle))
		}
		ctx, cancel := context.WithTimeout(context.Background(), callTimeout)
		defer cancel()
		return fn(ctx, m)
	})
}

// callID is call for entry points taking a JSON-encoded item id.
func callID(handle C.longlong, idJSON *C.char, fn func(ctx context.Context, m gomem.Memory, id string) (any, error)) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		var id string
		if err := json.Unmarshal([]byte(goStr(idJSON)), &id); err != nil {
			return nil, fmt.Errorf("id: %w", err)
		}
		return fn(ctx, m, id)
	})
}

// parseTime parses an RFC3339 timestamp (fractional seconds allowed); an
// empty or malformed one is the zero time.
func parseTime(s string) time.Time {
	t, _ := time.Parse(time.RFC3339, s)
	return t
}

func (p provDoc) toGo() gomem.Provenance {
	return gomem.Provenance{
		Source:       p.Source,
		RunID:        p.RunID,
		NodeRole:     p.NodeRole,
		Confidence:   p.Confidence,
		EvidenceRefs: p.EvidenceRefs,
		CreatedAt:    parseTime(p.CreatedAt),
	}
}

func toWireProv(p gomem.Provenance) provDoc {
	ca := ""
	if !p.CreatedAt.IsZero() {
		ca = p.CreatedAt.UTC().Format(time.RFC3339)
	}
	return provDoc{
		Source:       p.Source,
		RunID:        p.RunID,
		NodeRole:     p.NodeRole,
		Confidence:   p.Confidence,
		EvidenceRefs: p.EvidenceRefs,
		CreatedAt:    ca,
	}
}

// toGo converts a wire item to the public Item (SupersededBy is output only).
func (d itemDoc) toGo() gomem.Item {
	return gomem.Item{
		ID:         d.ID,
		Kind:       gomem.Kind(d.Kind),
		Title:      d.Title,
		Body:       d.Body,
		Payload:    d.Payload,
		Provenance: d.Provenance.toGo(),
		Tier:       d.Tier,
		Tags:       d.Tags,
		Supersedes: d.Supersedes,
	}
}

// toWireItem converts a public Item to its wire form.
func toWireItem(it gomem.Item) itemDoc {
	return itemDoc{
		ID:           it.ID,
		Kind:         string(it.Kind),
		Title:        it.Title,
		Body:         it.Body,
		Payload:      it.Payload,
		Provenance:   toWireProv(it.Provenance),
		Tier:         it.Tier,
		Tags:         it.Tags,
		Supersedes:   it.Supersedes,
		SupersededBy: it.SupersededBy,
	}
}

// ---------------------------------------------------------------------------
// Exported entry points
// ---------------------------------------------------------------------------

// GomemNew opens (or creates) a memory store.
//
//export GomemNew
func GomemNew(cfgJSON *C.char) *C.char {
	return run(func() (any, error) {
		cfg := gomem.DefaultConfig()
		var doc configDoc
		if raw := goStr(cfgJSON); raw != "" {
			if err := json.Unmarshal([]byte(raw), &doc); err != nil {
				return nil, fmt.Errorf("config: %w", err)
			}
			overlay(doc, &cfg)
		}
		cfg.Logger = newLogger(doc.LogLevel)
		ctx, cancel := context.WithTimeout(context.Background(), callTimeout)
		defer cancel()
		return openMemory(ctx, cfg)
	})
}

// openMemory wraps gomem.New for the bridge. A recovered-from-corrupt
// store is usable in Go as (Memory, ErrCorrupt); the wire form is
// {"handle": N, "error": "..."} so the client can keep the handle.
// Any other error Closes a non-nil Memory and returns no handle.
func openMemory(ctx context.Context, cfg gomem.Config) (any, error) {
	m, err := gomem.New(ctx, cfg)
	if m == nil {
		return nil, err
	}
	if err != nil && !errors.Is(err, gomem.ErrCorrupt) {
		_ = m.Close()
		return nil, err
	}
	out := map[string]any{"handle": register(m)}
	if err != nil {
		out["error"] = err.Error()
	}
	return out, nil
}

// GomemRemember persists an item; returns {"id": "<ulid>"}.
//
//export GomemRemember
func GomemRemember(handle C.longlong, itemJSON *C.char) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		var doc itemDoc
		if err := json.Unmarshal([]byte(goStr(itemJSON)), &doc); err != nil {
			return nil, fmt.Errorf("item: %w", err)
		}
		id, err := m.Remember(ctx, doc.toGo())
		if err != nil {
			return nil, err
		}
		return map[string]any{"id": id}, nil
	})
}

// GomemRememberMany persists an array of items atomically (all or none);
// returns {"ids": ["<ulid>", ...]} in input order.
//
//export GomemRememberMany
func GomemRememberMany(handle C.longlong, itemsJSON *C.char) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		return rememberMany(ctx, m, goStr(itemsJSON))
	})
}

func rememberMany(ctx context.Context, m gomem.Memory, itemsJSON string) (any, error) {
	var docs []itemDoc
	if err := json.Unmarshal([]byte(itemsJSON), &docs); err != nil {
		return nil, fmt.Errorf("items: %w", err)
	}
	items := make([]gomem.Item, len(docs))
	for i, d := range docs {
		items[i] = d.toGo()
	}
	ids, err := m.RememberMany(ctx, items)
	if err != nil {
		return nil, err
	}
	return map[string]any{"ids": ids}, nil
}

// GomemGet retrieves an item by id; returns the item document.
//
//export GomemGet
func GomemGet(handle C.longlong, idJSON *C.char) *C.char {
	return callID(handle, idJSON, func(ctx context.Context, m gomem.Memory, id string) (any, error) {
		it, err := m.Get(ctx, id)
		if err != nil {
			return nil, err
		}
		return toWireItem(it), nil
	})
}

// GomemDelete tombstones an item by id; returns {}.
//
//export GomemDelete
func GomemDelete(handle C.longlong, idJSON *C.char) *C.char {
	return callID(handle, idJSON, func(ctx context.Context, m gomem.Memory, id string) (any, error) {
		return struct{}{}, m.Delete(ctx, id)
	})
}

// GomemTouch records an access to an item by id; returns {}.
//
//export GomemTouch
func GomemTouch(handle C.longlong, idJSON *C.char) *C.char {
	return callID(handle, idJSON, func(ctx context.Context, m gomem.Memory, id string) (any, error) {
		return struct{}{}, m.Touch(ctx, id)
	})
}

// GomemRecall runs hybrid retrieval; returns {"hits": [...]}.
//
//export GomemRecall
func GomemRecall(handle C.longlong, queryJSON *C.char) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		var doc queryDoc
		if err := json.Unmarshal([]byte(goStr(queryJSON)), &doc); err != nil {
			return nil, fmt.Errorf("query: %w", err)
		}
		q := gomem.RecallQuery{
			Goal:                  doc.Goal,
			SubGoal:               doc.SubGoal,
			Kinds:                 make([]gomem.Kind, len(doc.Kinds)),
			MinTier:               doc.MinTier,
			Tags:                  doc.Tags,
			MinConfidence:         doc.MinConfidence,
			TopK:                  doc.TopK,
			Budget:                doc.BudgetTokens,
			RecencyHalfLifeDays:   doc.RecencyHalfLifeDays,
			IncludeSuperseded:     doc.IncludeSuperseded,
			DedupJaccardThreshold: doc.DedupJaccardThreshold,
		}
		for i, k := range doc.Kinds {
			q.Kinds[i] = gomem.Kind(k)
		}
		hits, err := m.Recall(ctx, q)
		if err != nil {
			return nil, err
		}
		wire := make([]hitDoc, len(hits))
		for i, h := range hits {
			wire[i] = hitDoc{
				ID:         h.ID,
				Kind:       string(h.Kind),
				Title:      h.Title,
				Body:       h.Body,
				Score:      h.Score,
				Tier:       h.Tier,
				Tokens:     h.Tokens,
				Provenance: toWireProv(h.Provenance),
			}
		}
		return map[string]any{"hits": wire}, nil
	})
}

// GomemReflect runs a housekeeping pass; returns the report.
//
//export GomemReflect
func GomemReflect(handle C.longlong, trigger *C.char) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		t := gomem.ReflectTrigger(parseTriggerArg(goStr(trigger)))
		if t == "" {
			t = gomem.TriggerRunEnd
		}
		rep, err := m.Reflect(ctx, t)
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"decayed":     rep.ItemsDecayed,
			"evicted":     rep.ItemsEvicted,
			"purged":      rep.ItemsPurged,
			"duration_ms": rep.DurationMs,
		}, nil
	})
}

// GomemStats returns the metrics snapshot.
//
//export GomemStats
func GomemStats(handle C.longlong) *C.char {
	return call(handle, func(ctx context.Context, m gomem.Memory) (any, error) {
		st, err := m.Stats(ctx)
		if err != nil {
			return nil, err
		}
		byKind := map[string]int{}
		for k, c := range st.ByKind {
			byKind[string(k)] = c
		}
		lastVacuum := ""
		if st.LastVacuumAt != nil {
			lastVacuum = st.LastVacuumAt.UTC().Format(time.RFC3339)
		}
		return map[string]any{
			"total_items":    st.TotalItems,
			"by_kind":        byKind,
			"by_tier":        st.ByTier,
			"db_size_bytes":  st.DBSizeBytes,
			"wal_size_bytes": st.WALSizeBytes,
			"last_vacuum_at": lastVacuum,
		}, nil
	})
}

// GomemClose flushes the WAL and releases the handle.
//
//export GomemClose
func GomemClose(handle C.longlong) *C.char {
	return run(func() (any, error) {
		m, ok := acquire(uintptr(handle))
		if !ok {
			return map[string]any{}, nil // idempotent
		}
		if err := m.Close(); err != nil {
			return nil, err
		}
		deregister(uintptr(handle))
		return map[string]any{}, nil
	})
}

// GomemFreeString releases a string returned by any entry point.
//
//export GomemFreeString
func GomemFreeString(s *C.char) {
	if s != nil {
		C.free(unsafe.Pointer(s))
	}
}

// ---------------------------------------------------------------------------
// Config overlay + logger
// ---------------------------------------------------------------------------

// overlay copies the fields present in doc onto cfg (leaving defaults
// intact for absent ones). An explicit zero is applied.
func overlay(doc configDoc, cfg *gomem.Config) {
	cfg.Path = cmp.Or(doc.Path, cfg.Path)
	cfg.Embed.Provider = cmp.Or(doc.Provider, cfg.Embed.Provider)
	cfg.Embed.APIURL = cmp.Or(doc.APIURL, cfg.Embed.APIURL)
	cfg.Embed.APIKey = cmp.Or(doc.APIKey, cfg.Embed.APIKey)
	cfg.Embed.Model = cmp.Or(doc.Model, cfg.Embed.Model)
	cfg.Embed.CacheDir = cmp.Or(doc.CacheDir, cfg.Embed.CacheDir)
	cfg.DenseIndex = cmp.Or(doc.DenseIndex, cfg.DenseIndex)
	cfg.FusionMode = cmp.Or(doc.FusionMode, cfg.FusionMode)
	if doc.TimeoutMS != nil {
		cfg.Embed.Timeout = time.Duration(*doc.TimeoutMS) * time.Millisecond
	}
	set(&cfg.Dim, doc.Dim)
	set(&cfg.Embed.MaxRetries, doc.MaxRetries)
	set(&cfg.Embed.BatchSize, doc.BatchSize)
	set(&cfg.DefaultTopK, doc.DefaultTopK)
	set(&cfg.RecencyHalfLifeDays, doc.RecencyHalfLifeDays)
	set(&cfg.DedupJaccardThreshold, doc.DedupJaccardThreshold)
	set(&cfg.AlphaDense, doc.AlphaDense)
	set(&cfg.BetaSparse, doc.BetaSparse)
	set(&cfg.GammaSalience, doc.GammaSalience)
	set(&cfg.WConfidence, doc.WConfidence)
	set(&cfg.WFrequency, doc.WFrequency)
	set(&cfg.WRecency, doc.WRecency)
	set(&cfg.FreqCap, doc.FreqCap)
	set(&cfg.EvictionAgeDays, doc.EvictionAgeDays)
	set(&cfg.MaxItems, doc.MaxItems)
	set(&cfg.MaxPayloadKB, doc.MaxPayloadKB)
	set(&cfg.EmbedLengthThreshold, doc.EmbedLengthThreshold)
	if doc.SendDimensions != nil {
		cfg.Embed.SendDimensions = doc.SendDimensions
	}
}

// set assigns *src to *dst when src is present.
func set[T any](dst, src *T) {
	if src != nil {
		*dst = *src
	}
}

// parseTriggerArg accepts either a bare trigger name or a JSON object
// carrying a "trigger" field.
func parseTriggerArg(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	if raw[0] == '{' {
		var doc struct {
			Trigger string `json:"trigger"`
		}
		if err := json.Unmarshal([]byte(raw), &doc); err != nil {
			return ""
		}
		return doc.Trigger
	}
	return raw
}

func newLogger(levelName string) *slog.Logger {
	var lvl slog.Level
	switch strings.ToLower(strings.TrimSpace(levelName)) {
	case "debug":
		lvl = slog.LevelDebug
	case "info":
		lvl = slog.LevelInfo
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelWarn
	}
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))
}

func main() {}
