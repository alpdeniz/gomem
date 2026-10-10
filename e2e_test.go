package gomem_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alpdeniz/gomem"
	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/testutil"
)

// Live embedding is opt-in so the default suite stays offline and fast:
//
//	GOMEM_TEST_LIVE_EMBED=1 go test -count=1 -timeout 3m -run TestE2E -v
//
// or `make test-e2e-live` (sources .env). The server URL and key come from
// GOMEM_EMBED_API_URL / GOMEM_EMBED_API_KEY. Optional:
// GOMEM_TEST_EMBED_PROVIDER (llama.cpp or openai), GOMEM_TEST_EMBED_MODEL.
func useLiveEmbed() bool {
	v := os.Getenv("GOMEM_TEST_LIVE_EMBED")
	return v != "" && v != "0"
}

var (
	liveOnce sync.Once
	liveErr  error
)

// e2eConfig opens on path with the mock embedder, or the real HTTP
// embedder when GOMEM_TEST_LIVE_EMBED is set. A live run probes the
// server once so a dead endpoint fails the test instead of silently
// degrading to keyword-only storage.
func e2eConfig(t *testing.T, path string) gomem.Config {
	t.Helper()
	cfg := testConfig()
	cfg.Path = path
	cfg.Logger = slog.New(slog.DiscardHandler)
	if !useLiveEmbed() {
		cfg.Embedder = testutil.NewMockEmbedder(testDim)
		return cfg
	}
	if p := os.Getenv("GOMEM_TEST_EMBED_PROVIDER"); p != "" {
		cfg.Embed.Provider = p
	}
	if m := os.Getenv("GOMEM_TEST_EMBED_MODEL"); m != "" {
		cfg.Embed.Model = m
	}
	liveOnce.Do(func() { liveErr = pingLiveEmbed(cfg) })
	if liveErr != nil {
		t.Fatalf("live embedder unreachable: %v\nset %s / %s and start the server, or unset GOMEM_TEST_LIVE_EMBED",
			liveErr, gomem.EnvAPIURL, gomem.EnvAPIKey)
	}
	return cfg
}

func pingLiveEmbed(cfg gomem.Config) error {
	url := cfg.Embed.APIURL
	if url == "" {
		url = os.Getenv(gomem.EnvAPIURL)
	}
	key := cfg.Embed.APIKey
	if key == "" {
		key = os.Getenv(gomem.EnvAPIKey)
	}
	provider := cfg.Embed.Provider
	if provider == "" {
		provider = "llama.cpp"
	}
	var emb gomem.Embedder
	switch strings.ToLower(provider) {
	case "openai", "api":
		emb = embed.NewAPIEmbedder(embed.APIConfig{
			APIURL: url, APIKey: key, Model: cfg.Embed.Model,
			Dim: cfg.Dim, Timeout: 15 * time.Second, MaxRetries: 1,
		})
	default:
		emb = embed.NewLlamaCppEmbedder(embed.LlamaCppConfig{
			ServerURL: url, APIKey: key, Model: cfg.Embed.Model,
			Dim: cfg.Dim, Timeout: 15 * time.Second, MaxRetries: 1,
		})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	vecs, err := emb.Embed(ctx, []string{"gomem e2e live probe"})
	if err != nil {
		return err
	}
	if len(vecs) != 1 || len(vecs[0]) == 0 {
		return fmt.Errorf("empty embedding from %s", provider)
	}
	return nil
}

// TestE2E drives one Memory handle through a full agent session and the
// edge cases that should not corrupt it. It uses only the public API.
func TestE2E(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "e2e.db")
	open := func() gomem.Memory {
		t.Helper()
		cfg := e2eConfig(t, path)
		m, err := gomem.New(ctx, cfg)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return m
	}
	m := open()
	defer func() { _ = m.Close() }()

	// ids is filled by the seed step and read by later steps.
	ids := map[string]string{}

	step := func(name string, fn func(*testing.T)) {
		t.Helper()
		if t.Failed() {
			return
		}
		t.Run(name, fn)
	}

	step("empty store", func(t *testing.T) {
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "anything at all", TopK: 10})
		testutil.AssertNoError(t, err)
		if len(hits) != 0 {
			t.Fatalf("expected no hits, got %d", len(hits))
		}
		st, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		if st.TotalItems != 0 {
			t.Fatalf("expected 0 items, got %d", st.TotalItems)
		}
	})

	step("reject invalid input", func(t *testing.T) {
		before, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		for name, item := range map[string]gomem.Item{
			"missing kind":     {Title: "no kind"},
			"missing title":    {Kind: gomem.KindFact},
			"whitespace title": {Kind: gomem.KindFact, Title: "   "},
			"whitespace id":    {ID: "   ", Kind: gomem.KindFact, Title: "t"},
			"self-supersede":   {ID: "self", Kind: gomem.KindFact, Title: "t", Supersedes: "self"},
			"superseded-by":    {Kind: gomem.KindFact, Title: "t", SupersededBy: "ghost"},
			"NaN confidence":   {Kind: gomem.KindFact, Title: "t", Provenance: gomem.Provenance{Confidence: gomem.Confidence(math.NaN())}},
			"unmarshalable":    {Kind: gomem.KindFact, Title: "t", Payload: make(chan int)},
			"empty batch item": {Kind: gomem.KindFact, Title: "ok"}, // paired below
		} {
			if name == "empty batch item" {
				_, err := m.RememberMany(ctx, []gomem.Item{item, {Kind: gomem.KindFact}})
				if !errors.Is(err, gomem.ErrInvalid) {
					t.Fatalf("%s: expected ErrInvalid, got %v", name, err)
				}
				continue
			}
			_, err := m.Remember(ctx, item)
			if !errors.Is(err, gomem.ErrInvalid) {
				t.Fatalf("%s: expected ErrInvalid, got %v", name, err)
			}
		}
		after, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		if after.TotalItems != before.TotalItems {
			t.Fatalf("invalid writes leaked: %d → %d items", before.TotalItems, after.TotalItems)
		}
	})

	step("seed corpus", func(t *testing.T) {
		corpus := []gomem.Item{
			{
				ID: "idx", Kind: gomem.KindFact,
				Title: "Composite indexes accelerate filtered database queries",
				Body:  "Aligning a composite index with WHERE and ORDER BY removes table scans and sorts.",
				Tags:  []string{"Database", " QUERY ", "database", "index"},
				Provenance: gomem.Provenance{
					Source: "run", RunID: "e2e",
					Confidence: gomem.Confidence(0.95),
				},
			},
			{
				ID: "idx-dup", Kind: gomem.KindFact,
				Title:      "Composite indexes accelerate filtered database queries too",
				Body:       "Aligning a composite index with WHERE and ORDER BY removes table scans and sorts.",
				Tags:       []string{"database", "index"},
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.9)},
			},
			{
				ID: "cache", Kind: gomem.KindLesson,
				Title:      "Cache hot reads to eliminate redundant computation",
				Body:       "Memoizing frequently read, slowly changing results lowers latency and CPU spend.",
				Tags:       []string{"cache", "performance"},
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.85)},
			},
			{
				ID: "profile", Kind: gomem.KindSkill,
				Title:      "Profile before you optimize",
				Body:       "Use a profiler to find the real hotspot; do not guess.",
				Tags:       []string{"profiling"},
				Provenance: gomem.Provenance{Source: "human", Confidence: gomem.Confidence(0.7)},
			},
			{
				ID: "auth", Kind: gomem.KindFact,
				Title:      "Authentication tokens should expire and rotate",
				Body:       "Short-lived bearer tokens with rotation limit the blast radius of a leak.",
				Tags:       []string{"auth", "token"},
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.9)},
			},
			{
				ID: "rumor", Kind: gomem.KindFact,
				Title:      "Rumored speedup from disabling indexes",
				Body:       "Unverified claim that dropping every index makes writes faster.",
				Tags:       []string{"database"},
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0)},
			},
			{
				ID: "cold", Kind: gomem.KindFact,
				Title:      "Legacy connection-pool sizing notes",
				Body:       "An old note about bounding the number of open database connections.",
				Tags:       []string{"pool"},
				Tier:       gomem.Tier(0),
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.8)},
			},
			{
				ID: "short", Kind: gomem.KindFact,
				Title:      "Xyz", // 3 runes: below EmbedLengthThreshold, so FTS-only
				Body:       "",
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.8)},
			},
			{
				ID: "playbook", Kind: gomem.Kind("playbook"),
				Title: "On-call playbook for replica lag",
				Body:  "Check replication slots, then pause writers if apply lag exceeds five minutes.",
				Tags:  []string{"ops"},
				Payload: map[string]any{
					"severity": "high",
					"steps":    3.0,
					"ok":       true,
				},
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.88)},
			},
			{
				ID: "rare", Kind: gomem.KindFact,
				Title:      "Zebracorn pasture calendar",
				Body:       "Zebracorn herds migrate when the qwertyuiop grass dries out.",
				Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.8)},
			},
		}
		got, err := m.RememberMany(ctx, corpus)
		testutil.AssertNoError(t, err)
		if len(got) != len(corpus) {
			t.Fatalf("RememberMany returned %d ids, want %d", len(got), len(corpus))
		}
		for i, it := range corpus {
			if got[i] != it.ID {
				t.Fatalf("id[%d] = %q, want %q", i, got[i], it.ID)
			}
			ids[it.ID] = got[i]
		}
		empty, err := m.RememberMany(ctx, nil)
		testutil.AssertNoError(t, err)
		if len(empty) != 0 {
			t.Fatalf("empty batch returned %v", empty)
		}
	})

	step("hybrid recall ranks the relevant item", func(t *testing.T) {
		hits, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "speed up slow database query with indexes",
			TopK: 8,
		})
		testutil.AssertNoError(t, err)
		if len(hits) == 0 {
			t.Fatal("expected hits")
		}
		testutil.AssertSortedDesc(t, hits)
		testutil.AssertScoreRange(t, hits, 0, 1)
		if hits[0].ID != "idx" && hits[0].ID != "idx-dup" {
			t.Fatalf("expected an index item on top, got %q", hits[0].ID)
		}
		for _, h := range hits {
			if h.ID == "" || h.Title == "" || h.Provenance.Confidence == nil {
				t.Fatalf("hit missing required fields: %+v", h)
			}
			if h.Tokens <= 0 && (h.Title != "" || h.Body != "") {
				t.Fatalf("hit %s: tokens = %d", h.ID, h.Tokens)
			}
		}
	})

	step("filters are AND and can match nothing", func(t *testing.T) {
		hits, err := m.Recall(ctx, gomem.RecallQuery{
			Goal:          "authentication token expiry",
			Kinds:         []gomem.Kind{gomem.KindFact},
			Tags:          []string{"auth"},
			MinConfidence: 0.8,
			TopK:          10,
		})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, hits, "auth")
		for _, h := range hits {
			if h.Kind != gomem.KindFact {
				t.Fatalf("kind filter violated: %s", h.Kind)
			}
			if *h.Provenance.Confidence < 0.8 {
				t.Fatalf("confidence filter violated: %v", *h.Provenance.Confidence)
			}
		}

		hot, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "connection pool database", MinTier: 2, TopK: 20,
		})
		testutil.AssertNoError(t, err)
		for _, h := range hot {
			if h.ID == "cold" {
				t.Fatal("cold item passed MinTier 2")
			}
		}

		trusted, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "disabling indexes speedup", MinConfidence: 0.5, TopK: 20,
		})
		testutil.AssertNoError(t, err)
		for _, h := range trusted {
			if h.ID == "rumor" {
				t.Fatal("untrusted item passed MinConfidence 0.5")
			}
		}

		none, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "database query", Kinds: []gomem.Kind{gomem.KindTemplate}, TopK: 10,
		})
		testutil.AssertNoError(t, err)
		if len(none) != 0 {
			t.Fatalf("expected no template hits, got %d", len(none))
		}

		custom, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "replica lag playbook", Kinds: []gomem.Kind{"playbook"}, TopK: 5,
		})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, custom, "playbook")
	})

	step("shaping: TopK, budget, empty goal, SubGoal, dedup", func(t *testing.T) {
		top1, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "speed up slow database query with indexes", TopK: 1,
		})
		testutil.AssertNoError(t, err)
		if len(top1) != 1 {
			t.Fatalf("TopK=1 returned %d hits", len(top1))
		}

		capped, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "speed up slow database query with indexes", TopK: 10, Budget: 1,
		})
		testutil.AssertNoError(t, err)
		if len(capped) != 1 {
			t.Fatalf("Budget=1 should keep only the first hit, got %d", len(capped))
		}

		emptyGoal, err := m.Recall(ctx, gomem.RecallQuery{Goal: "", TopK: 5})
		testutil.AssertNoError(t, err)
		if len(emptyGoal) != 0 {
			t.Fatalf("empty goal: expected no hits, got %d", len(emptyGoal))
		}

		// Goal is off-topic; SubGoal supplies the sparse terms that find the rare item.
		wide, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "cooking pasta", SubGoal: "zebracorn qwertyuiop", TopK: 10,
		})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, wide, "rare")

		deduped, err := m.Recall(ctx, gomem.RecallQuery{
			Goal:                  "composite indexes accelerate filtered database queries",
			TopK:                  10,
			DedupJaccardThreshold: 0.7,
		})
		testutil.AssertNoError(t, err)
		sawIdx, sawDup := false, false
		for _, h := range deduped {
			sawIdx = sawIdx || h.ID == "idx"
			sawDup = sawDup || h.ID == "idx-dup"
		}
		if sawIdx && sawDup {
			t.Fatal("near-duplicate index items were not collapsed")
		}
	})

	step("Get, Touch, and missing IDs", func(t *testing.T) {
		it, err := m.Get(ctx, "playbook")
		testutil.AssertNoError(t, err)
		if it.Kind != "playbook" || it.Title != "On-call playbook for replica lag" {
			t.Fatalf("Get(playbook) = %+v", it)
		}
		if it.Tier == nil || *it.Tier != 2 {
			t.Fatalf("new item tier = %v, want hot (2)", it.Tier)
		}
		if it.Provenance.Confidence == nil || *it.Provenance.Confidence != 0.88 {
			t.Fatalf("confidence = %v", it.Provenance.Confidence)
		}
		if !reflect.DeepEqual(it.Tags, []string{"ops"}) {
			t.Fatalf("tags = %v (input mixed-case/dupes should be cleaned)", it.Tags)
		}
		gotPayload, err := json.Marshal(it.Payload)
		testutil.AssertNoError(t, err)
		var payload map[string]any
		if err := json.Unmarshal(gotPayload, &payload); err != nil {
			t.Fatalf("payload: %v", err)
		}
		if payload["severity"] != "high" || payload["ok"] != true || payload["steps"] != 3.0 {
			t.Fatalf("payload = %#v", payload)
		}

		idx, err := m.Get(ctx, "idx")
		testutil.AssertNoError(t, err)
		// Tags are lowercased, de-duplicated, and stored in sorted order.
		if !reflect.DeepEqual(idx.Tags, []string{"database", "index", "query"}) {
			t.Fatalf("idx tags = %v", idx.Tags)
		}

		short, err := m.Get(ctx, "short")
		testutil.AssertNoError(t, err)
		if short.Title != "Xyz" {
			t.Fatalf("short title = %q", short.Title)
		}
		shortHits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "Xyz", TopK: 20})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, shortHits, "short")

		testutil.AssertNoError(t, m.Touch(ctx, "idx"))
		if _, err := m.Get(ctx, "ghost"); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("Get(ghost): %v", err)
		}
		if err := m.Touch(ctx, "ghost"); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("Touch(ghost): %v", err)
		}
		if err := m.Delete(ctx, "ghost"); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("Delete(ghost): %v", err)
		}
	})

	step("supersession hides the old item", func(t *testing.T) {
		newID, err := m.Remember(ctx, gomem.Item{
			ID: "idx-v2", Kind: gomem.KindFact,
			Title:      "Updated composite-index guidance",
			Body:       "Prefer a covering composite index that matches WHERE and ORDER BY.",
			Supersedes: "idx",
			Tags:       []string{"database", "index"},
			Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.97)},
		})
		testutil.AssertNoError(t, err)
		ids["idx-v2"] = newID

		hits, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "composite index database query", TopK: 20,
		})
		testutil.AssertNoError(t, err)
		for _, h := range hits {
			if h.ID == "idx" {
				t.Fatal("superseded idx should be hidden by default")
			}
		}
		testutil.AssertContainsID(t, hits, "idx-v2")

		withOld, err := m.Recall(ctx, gomem.RecallQuery{
			Goal: "composite index database query", TopK: 20,
			IncludeSuperseded:     true,
			DedupJaccardThreshold: 1, // don't collapse the superseded item into its replacement
		})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, withOld, "idx")

		old, err := m.Get(ctx, "idx")
		testutil.AssertNoError(t, err)
		if old.SupersededBy != "idx-v2" {
			t.Fatalf("idx.SupersededBy = %q", old.SupersededBy)
		}
		neu, err := m.Get(ctx, "idx-v2")
		testutil.AssertNoError(t, err)
		if neu.Supersedes != "idx" {
			t.Fatalf("idx-v2.Supersedes = %q", neu.Supersedes)
		}
		if _, err := m.Remember(ctx, neu); err != nil {
			t.Fatalf("round-trip Remember of superseded-by output: %v", err)
		}
	})

	step("delete, exclude, revive", func(t *testing.T) {
		testutil.AssertNoError(t, m.Delete(ctx, "cache"))
		if _, err := m.Get(ctx, "cache"); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("Get after delete: %v", err)
		}
		if err := m.Delete(ctx, "cache"); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("re-delete: %v", err)
		}
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "cache hot reads latency", TopK: 10})
		testutil.AssertNoError(t, err)
		for _, h := range hits {
			if h.ID == "cache" {
				t.Fatal("tombstone still recalled")
			}
		}

		_, err = m.Remember(ctx, gomem.Item{
			ID: "cache", Kind: gomem.KindLesson,
			Title:      "Cache hot reads to eliminate redundant computation",
			Body:       "Memoizing frequently read, slowly changing results lowers latency and CPU spend.",
			Tags:       []string{"cache", "performance"},
			Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.85)},
		})
		testutil.AssertNoError(t, err)
		if _, err := m.Get(ctx, "cache"); err != nil {
			t.Fatalf("revived cache: %v", err)
		}
		revived, err := m.Recall(ctx, gomem.RecallQuery{Goal: "cache hot reads latency", TopK: 10})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, revived, "cache")
	})

	step("idempotent re-remember", func(t *testing.T) {
		before, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		auth, err := m.Get(ctx, "auth")
		testutil.AssertNoError(t, err)
		id, err := m.Remember(ctx, auth)
		testutil.AssertNoError(t, err)
		if id != "auth" {
			t.Fatalf("id = %q", id)
		}
		after, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		if after.TotalItems != before.TotalItems {
			t.Fatalf("idempotent write changed count %d → %d", before.TotalItems, after.TotalItems)
		}
	})

	step("stats match the live inventory", func(t *testing.T) {
		st, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		if st.TotalItems <= 0 {
			t.Fatal("expected live items")
		}
		if st.ByKind[gomem.KindFact] == 0 || st.ByKind["playbook"] != 1 {
			t.Fatalf("ByKind = %v", st.ByKind)
		}
		if st.ByTier[0] < 1 || st.ByTier[2] < 1 {
			t.Fatalf("ByTier = %v", st.ByTier)
		}
		if st.DBSizeBytes <= 0 {
			t.Fatalf("DBSizeBytes = %d", st.DBSizeBytes)
		}
		sum := 0
		for _, n := range st.ByKind {
			sum += n
		}
		if sum != st.TotalItems {
			t.Fatalf("ByKind sum %d != TotalItems %d", sum, st.TotalItems)
		}
	})

	step("survives close and reopen", func(t *testing.T) {
		st, err := m.Stats(ctx)
		testutil.AssertNoError(t, err)
		want := st.TotalItems
		testutil.AssertNoError(t, m.Close())
		testutil.AssertNoError(t, m.Close()) // idempotent

		m = open()
		st, err = m.Stats(ctx)
		testutil.AssertNoError(t, err)
		if st.TotalItems != want {
			t.Fatalf("after reopen: %d items, want %d", st.TotalItems, want)
		}
		got, err := m.Get(ctx, "playbook")
		testutil.AssertNoError(t, err)
		if got.Title != "On-call playbook for replica lag" {
			t.Fatalf("playbook lost after reopen: %+v", got)
		}
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "composite index database", TopK: 5})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, hits, "idx-v2")
	})

	step("closed handle", func(t *testing.T) {
		testutil.AssertNoError(t, m.Close())
		if _, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "after close"}); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Remember: %v", err)
		}
		if _, err := m.Recall(ctx, gomem.RecallQuery{Goal: "x"}); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Recall: %v", err)
		}
		if _, err := m.Get(ctx, "idx"); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Get: %v", err)
		}
		if err := m.Delete(ctx, "idx"); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Delete: %v", err)
		}
		if err := m.Touch(ctx, "idx"); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Touch: %v", err)
		}
		if _, err := m.Stats(ctx); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Stats: %v", err)
		}
		if _, err := m.Reflect(ctx, gomem.TriggerRunEnd); !errors.Is(err, gomem.ErrClosed) {
			t.Fatalf("Reflect: %v", err)
		}
		testutil.AssertNoError(t, m.Close())
	})

	t.Run("capacity eviction keeps the newest", func(t *testing.T) {
		cfg := e2eConfig(t, filepath.Join(t.TempDir(), "cap.db"))
		cfg.MaxItems = 2
		cap, err := gomem.New(ctx, cfg)
		testutil.AssertNoError(t, err)
		defer cap.Close()

		var stored []string
		for i := 0; i < 4; i++ {
			id, err := cap.Remember(ctx, gomem.Item{
				ID:    fmt.Sprintf("cap-%d", i),
				Kind:  gomem.KindFact,
				Title: fmt.Sprintf("capacity marker variant %d", i),
				Body:  fmt.Sprintf("distinct eviction payload body %d exercising the ordering path", i),
				Provenance: gomem.Provenance{
					Source:     "run",
					Confidence: gomem.Confidence(0.7),
				},
			})
			testutil.AssertNoError(t, err)
			stored = append(stored, id)
		}
		// Touch raises salience (frequency + recency) so the cap evicts
		// the untouched oldest items, not an arbitrary tie.
		testutil.AssertNoError(t, cap.Touch(ctx, stored[2]))
		testutil.AssertNoError(t, cap.Touch(ctx, stored[3]))
		testutil.AssertNoError(t, cap.Touch(ctx, stored[3]))
		rep, err := cap.Reflect(ctx, gomem.TriggerPeriodic)
		testutil.AssertNoError(t, err)
		if rep == nil || rep.ItemsEvicted < 2 {
			t.Fatalf("report = %+v", rep)
		}
		st, err := cap.Stats(ctx)
		testutil.AssertNoError(t, err)
		if st.TotalItems > 2 {
			t.Fatalf("live items = %d, want <= 2", st.TotalItems)
		}
		if _, err := cap.Get(ctx, stored[3]); err != nil {
			t.Fatalf("highest-salience item missing: %v", err)
		}
		if _, err := cap.Get(ctx, stored[0]); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("lowest-salience item still live: %v", err)
		}
	})

	if useLiveEmbed() {
		t.Run("live semantic ranking", func(t *testing.T) {
			// A neighbourhood an agent would actually have: the right
			// memory, on-topic near-misses, a keyword collision that
			// means something else, and off-topic items. Confidence is
			// equal so salience cannot pick a winner. The first goal
			// shares no content words with the target, so FTS-only
			// fallback cannot retrieve it.
			cfg := e2eConfig(t, filepath.Join(t.TempDir(), "sem.db"))
			sem, err := gomem.New(ctx, cfg)
			testutil.AssertNoError(t, err)
			defer sem.Close()

			item := func(id, title, body string) gomem.Item {
				return gomem.Item{
					ID: id, Kind: gomem.KindFact, Title: title, Body: body,
					Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.8)},
				}
			}
			_, err = sem.RememberMany(ctx, []gomem.Item{
				item("scan", "Covering keys spare the heap",
					"When every column the planner needs already lives in the secondary structure, the engine never visits the base rows."),
				item("magnet", "SELECT the latest order for a result set",
					"Restrict the query to one row when you only need the most recent order."),
				item("explain", "Read the query plan before tuning",
					"EXPLAIN shows whether the planner picked a sequential walk or a key lookup."),
				item("drop", "Drop unused indexes to speed writes",
					"Each extra index slows inserts; remove the ones the planner never chooses."),
				item("pool", "Bound the connection pool",
					"A fixed number of sessions stops a burst of clients from exhausting the server."),
				item("cache", "Memoize hot reads",
					"Keep slowly changing answers in memory so repeated requests skip the store."),
				item("nplus", "Batch the child rows",
					"Loading related records one at a time inside a loop issues N plus one statements."),
				item("page", "Keyset pagination stays fast",
					"Deep OFFSET pages get slower; walk an indexed column instead."),
				item("auth", "Rotate bearer credentials",
					"Short-lived tokens limit the blast radius if a secret leaks."),
				item("bread", "Feed the sourdough starter twice a day",
					"Equal parts flour and water keep the culture active overnight."),
				item("hike", "Layer clothing for alpine weather swings",
					"A base layer, fleece and shell handle temperature drops at elevation."),
			})
			testutil.AssertNoError(t, err)

			rankOf := func(hits []gomem.Hit, id string) int {
				for i, h := range hits {
					if h.ID == id {
						return i
					}
				}
				return -1
			}
			idsOf := func(hits []gomem.Hit) []string {
				out := make([]string, len(hits))
				for i, h := range hits {
					out[i] = h.ID
				}
				return out
			}
			for _, goal := range []string{
				"Why is my SELECT so slow when I restrict and order the result set?",
				"I filter and sort but it still touches the clustered table",
			} {
				hits, err := sem.Recall(ctx, gomem.RecallQuery{
					Goal: goal, TopK: 5, DedupJaccardThreshold: 1,
				})
				testutil.AssertNoError(t, err)
				got := rankOf(hits, "scan")
				if got < 0 {
					t.Fatalf("%q: covering-key item not in top %d (got %v)", goal, 5, idsOf(hits))
				}
				for _, decoy := range []string{"bread", "hike", "auth"} {
					if d := rankOf(hits, decoy); d >= 0 && d < got {
						t.Fatalf("%q: off-topic %s ranked above the semantic match (%d < %d); hits=%v",
							goal, decoy, d, got, idsOf(hits))
					}
				}
			}
		})
	}

	t.Run("offline embedder and cancelled context", func(t *testing.T) {
		cfg := testConfig()
		cfg.Path = filepath.Join(t.TempDir(), "off.db")
		cfg.Logger = slog.New(slog.DiscardHandler)
		cfg.Embed.Provider = "offline"
		cfg.Embedder = nil
		off, err := gomem.New(ctx, cfg)
		testutil.AssertNoError(t, err)
		defer off.Close()

		id, err := off.Remember(ctx, gomem.Item{
			Kind:       gomem.KindFact,
			Title:      "offline stored fact",
			Body:       "written without any network embedding service",
			Provenance: gomem.Provenance{Confidence: gomem.Confidence(0.9)},
		})
		testutil.AssertNoError(t, err)
		hits, err := off.Recall(ctx, gomem.RecallQuery{Goal: "offline stored fact", TopK: 3})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, hits, id)

		canceled, cancel := context.WithCancel(ctx)
		cancel()
		if _, err := off.Recall(canceled, gomem.RecallQuery{Goal: "offline stored fact", TopK: 3}); err == nil {
			t.Fatal("Recall on a cancelled context succeeded")
		}
	})

	t.Run("oversized payload", func(t *testing.T) {
		cfg := e2eConfig(t, filepath.Join(t.TempDir(), "pay.db"))
		cfg.MaxPayloadKB = 1
		pay, err := gomem.New(ctx, cfg)
		testutil.AssertNoError(t, err)
		defer pay.Close()

		_, err = pay.Remember(ctx, gomem.Item{
			Kind:    gomem.KindFact,
			Title:   "huge",
			Payload: strings.Repeat("x", 2048),
		})
		if !errors.Is(err, gomem.ErrInvalid) {
			t.Fatalf("expected ErrInvalid, got %v", err)
		}
		st, err := pay.Stats(ctx)
		testutil.AssertNoError(t, err)
		if st.TotalItems != 0 {
			t.Fatalf("oversized payload was stored")
		}
	})
}
