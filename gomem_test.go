package gomem_test

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	_ "modernc.org/sqlite"

	"github.com/alpdeniz/gomem"
	"github.com/alpdeniz/gomem/internal/testutil"
)

const testDim = 512

// testConfig is DefaultConfig with the dense index chosen by
// GOMEM_TEST_DENSE_INDEX, so the whole suite can run against either mode:
//
//	GOMEM_TEST_DENSE_INDEX=memory go test ./...
func testConfig() gomem.Config {
	cfg := gomem.DefaultConfig()
	if m := os.Getenv("GOMEM_TEST_DENSE_INDEX"); m != "" {
		cfg.DenseIndex = m
	}
	return cfg
}

// newTestMemory opens a Memory on a temp file using the supplied
// embedder and registers cleanup.
func newTestMemory(t *testing.T, emb interface {
	Embed(context.Context, []string) ([][]float32, error)
	Dim() int
}) (gomem.Memory, func()) {
	t.Helper()
	dir := t.TempDir()
	cfg := testConfig()
	cfg.Path = filepath.Join(dir, "test.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = emb
	m, err := gomem.New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return m, func() { m.Close() }
}

func TestReflect_UnknownTriggerIsInvalid(t *testing.T) {
	m, cleanup := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer cleanup()
	for _, tr := range []gomem.ReflectTrigger{"", "periodc"} {
		if _, err := m.Reflect(context.Background(), tr); !errors.Is(err, gomem.ErrInvalid) {
			t.Fatalf("Reflect(%q): want ErrInvalid, got %v", tr, err)
		}
	}
}

// seedItems remembers a small curated corpus and returns title->ID.
func seedItems(t *testing.T, ctx context.Context, m gomem.Memory) map[string]string {
	t.Helper()
	corpus := []gomem.Item{
		{
			Kind:  gomem.KindFact,
			Title: "Composite indexes accelerate filtered database queries",
			Body:  "Aligning a composite index with WHERE and ORDER BY removes table scans and sorts, cutting query latency.",
			Tags:  []string{"database", "query", "index"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "seed",
				Confidence: gomem.Confidence(0.95),
			},
		},
		{
			Kind:  gomem.KindLesson,
			Title: "Cache hot reads to eliminate redundant computation",
			Body:  "Memoizing frequently read, slowly changing results lowers latency and CPU spend.",
			Tags:  []string{"cache", "performance"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "seed",
				Confidence: gomem.Confidence(0.85),
			},
		},
		{
			Kind:  gomem.KindSkill,
			Title: "Profile before you optimize",
			Body:  "Use a profiler to find the real hotspot; do not guess.",
			Tags:  []string{"profiling"},
			Provenance: gomem.Provenance{
				Source:     "human",
				Confidence: gomem.Confidence(0.7),
			},
		},
		{
			Kind:  gomem.KindFact,
			Title: "Authentication tokens should expire and rotate",
			Body:  "Short-lived bearer tokens with rotation limit the blast radius of a leak.",
			Tags:  []string{"auth", "token"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "seed",
				Confidence: gomem.Confidence(0.9),
			},
		},
	}
	ids := map[string]string{}
	for i := range corpus {
		id, err := m.Remember(ctx, corpus[i])
		if err != nil {
			t.Fatalf("Remember[%d]: %v", i, err)
		}
		ids[corpus[i].Title] = id
	}
	return ids
}

// TestRecall_HappyPath covers the hybrid recall happy path: ranked,
// normalized, descending results with relevant items on top.
func TestRecall_HappyPath(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()
	seedItems(t, ctx, m)

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "speed up slow database query with indexes",
		TopK: 5,
	})
	testutil.AssertNoError(t, err)
	if len(hits) == 0 {
		t.Fatal("expected hits, got none")
	}
	testutil.AssertSortedDesc(t, hits)
	testutil.AssertScoreRange(t, hits, 0, 1)

	top := hits[0]
	if !containsAny(top.Title, "index", "quer") {
		t.Fatalf("expected a database/index item on top, got %q", top.Title)
	}
}

// TestRecall_Filters verifies AND semantics across kinds, tags, and
// minimum confidence.
func TestRecall_Filters(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()
	ids := seedItems(t, ctx, m)

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal:          "authentication token expiry",
		Kinds:         []gomem.Kind{gomem.KindFact},
		Tags:          []string{"auth"},
		MinConfidence: 0.8,
		TopK:          10,
	})
	testutil.AssertNoError(t, err)
	for _, h := range hits {
		if h.Kind != gomem.KindFact {
			t.Fatalf("kind filter violated: %s", h.Kind)
		}
		if *h.Provenance.Confidence < 0.8 {
			t.Fatalf("confidence filter violated: %v", *h.Provenance.Confidence)
		}
	}
	testutil.AssertContainsID(t, hits, ids["Authentication tokens should expire and rotate"])
}

// TestRecall_EmbedderDown_Degrades confirms a failing embedder degrades
// to FTS-only results without surfacing an error.
func TestRecall_EmbedderDown_Degrades(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewErrEmbedder(testDim))
	defer done()
	seedItems(t, ctx, m)

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "composite index database query",
		TopK: 5,
	})
	testutil.AssertNoError(t, err)
	if len(hits) == 0 {
		t.Fatal("expected FTS-only hits even with the embedder down")
	}
	testutil.AssertScoreRange(t, hits, 0, 1)
}

// TestRecall_BothLegsEmpty covers a brand-new database.
func TestRecall_BothLegsEmpty(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "anything at all",
		TopK: 10,
	})
	testutil.AssertNoError(t, err)
	if len(hits) != 0 {
		t.Fatalf("expected no hits on an empty DB, got %d", len(hits))
	}
}

// TestRemember_New verifies atomic persistence and ULID return.
func TestRemember_New(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "A brand new memorable fact",
		Body:  "It talks about something quite specific and searchable.",
		Tags:  []string{"unique"},
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	id, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)
	if len(id) != 26 {
		t.Fatalf("expected a 26-char ULID, got %q (len %d)", id, len(id))
	}

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "brand new memorable fact specific searchable",
		TopK: 3,
	})
	testutil.AssertNoError(t, err)
	testutil.AssertContainsID(t, hits, id)
}

// TestRemember_Idempotent confirms the same ID + content is a no-op.
func TestRemember_Idempotent(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		ID:    "fixed-id-0001",
		Kind:  gomem.KindFact,
		Title: "Stable content",
		Body:  "Unchanged body",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.8),
		},
	}
	first, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)
	second, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)
	if first != second || first != "fixed-id-0001" {
		t.Fatalf("idempotency broken: %q vs %q", first, second)
	}
	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems != 1 {
		t.Fatalf("expected 1 item after idempotent double-write, got %d", st.TotalItems)
	}
}

// TestRemember_Supersedes checks the supersession link and the
// recall-time penalty/exclusion.
func TestRemember_Supersedes(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	old := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "Outdated guideline about caching",
		Body:  "Old advice regarding cache invalidation strategies.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.6),
		},
	}
	oldID, err := m.Remember(ctx, old)
	testutil.AssertNoError(t, err)

	new := gomem.Item{
		Kind:       gomem.KindFact,
		Title:      "Updated guideline about caching",
		Body:       "Refined advice on cache invalidation and TTL policies.",
		Supersedes: oldID,
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	newID, err := m.Remember(ctx, new)
	testutil.AssertNoError(t, err)
	if newID == "" {
		t.Fatal("expected a new ID for the superseding item")
	}

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "guideline about caching invalidation",
		TopK: 10,
	})
	testutil.AssertNoError(t, err)
	for _, h := range hits {
		if h.ID == oldID {
			t.Fatalf("superseded item %s should be excluded by default", oldID)
		}
	}
	testutil.AssertContainsID(t, hits, newID)

	withOld, err := m.Recall(ctx, gomem.RecallQuery{
		Goal:              "guideline about caching invalidation",
		TopK:              10,
		IncludeSuperseded: true,
	})
	testutil.AssertNoError(t, err)
	testutil.AssertContainsID(t, withOld, oldID)
}

// TestGet_SupersessionRoundTrip confirms Get reports both directions of a
// supersession link and that re-saving a fetched item leaves it intact.
func TestGet_SupersessionRoundTrip(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	oldID, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "Old", Body: "old body"})
	testutil.AssertNoError(t, err)
	newID, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "New", Body: "new body", Supersedes: oldID})
	testutil.AssertNoError(t, err)

	newItem, err := m.Get(ctx, newID)
	testutil.AssertNoError(t, err)
	if newItem.Supersedes != oldID || newItem.SupersededBy != "" {
		t.Fatalf("new item links = (%q, %q), want (%q, \"\")", newItem.Supersedes, newItem.SupersededBy, oldID)
	}
	oldItem, err := m.Get(ctx, oldID)
	testutil.AssertNoError(t, err)
	if oldItem.SupersededBy != newID || oldItem.Supersedes != "" {
		t.Fatalf("old item links = (%q, %q), want (\"\", %q)", oldItem.Supersedes, oldItem.SupersededBy, newID)
	}

	for _, it := range []gomem.Item{newItem, oldItem} {
		if _, err := m.Remember(ctx, it); err != nil {
			t.Fatalf("round-trip Remember(%s): %v", it.ID, err)
		}
	}
	newAgain, _ := m.Get(ctx, newID)
	oldAgain, _ := m.Get(ctx, oldID)
	if newAgain.Supersedes != oldID || oldAgain.SupersededBy != newID {
		t.Fatalf("links changed after round trip: new=%+v old=%+v", newAgain, oldAgain)
	}
}

// TestRemember_SupersededByOnInputRejected confirms the read-only field
// fails loudly instead of being silently ignored.
func TestRemember_SupersededByOnInputRejected(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	oldID, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "Old", Body: "old body"})
	testutil.AssertNoError(t, err)
	_, err = m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "New", Body: "new body", SupersededBy: oldID})
	if !errors.Is(err, gomem.ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}

// TestRemember_SameContentNewLinkIsWritten confirms re-remembering
// unchanged content with a new supersession link still writes the link.
func TestRemember_SameContentNewLinkIsWritten(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	oldID, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "Old", Body: "old body"})
	testutil.AssertNoError(t, err)
	it := gomem.Item{
		Kind: gomem.KindFact, Title: "New", Body: "new body",
		Provenance: gomem.Provenance{Source: "run", Confidence: gomem.Confidence(1), CreatedAt: time.Now().UTC().Truncate(time.Second)},
	}
	newID, err := m.Remember(ctx, it)
	testutil.AssertNoError(t, err)

	it.ID, it.Supersedes = newID, oldID
	_, err = m.Remember(ctx, it)
	testutil.AssertNoError(t, err)
	oldItem, err := m.Get(ctx, oldID)
	testutil.AssertNoError(t, err)
	if oldItem.SupersededBy != newID {
		t.Fatalf("SupersededBy = %q, want %q", oldItem.SupersededBy, newID)
	}
}

// TestRemember_ValidationFailure covers empty kind/title.
func TestRemember_ValidationFailure(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	_, err := m.Remember(ctx, gomem.Item{Title: "no kind"})
	testutil.AssertError(t, err)
	if !errors.Is(err, gomem.ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
	_, err = m.Remember(ctx, gomem.Item{Kind: gomem.KindFact})
	testutil.AssertError(t, err)
	if !errors.Is(err, gomem.ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}

// TestSoftDelete_ExcludedFromRecall covers the tombstone behaviour.
func TestSoftDelete_ExcludedFromRecall(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "softdel.db")
	cfg := testConfig()
	cfg.Path = path
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	item := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "A deletable fact about networking",
		Body:  "Details on socket timeouts and reconnect backoff.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	id, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)

	before, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "socket timeouts reconnect backoff networking",
		TopK: 5,
	})
	testutil.AssertNoError(t, err)
	testutil.AssertContainsID(t, before, id)

	// Soft-delete by flipping deleted_at through a raw connection
	// (the public API defers deletion to Reflect's eviction in v1).
	testutil.AssertNoError(t, tombstoneRow(path, id))

	after, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "socket timeouts reconnect backoff networking",
		TopK: 5,
	})
	testutil.AssertNoError(t, err)
	for _, h := range after {
		if h.ID == id {
			t.Fatalf("tombstoned item %s should be excluded from recall", id)
		}
	}
}

// tombstoneRow opens the SQLite file directly and sets deleted_at on
// the given item, exercising the soft-delete path.
func tombstoneRow(path, id string) error {
	conn, err := sql.Open("sqlite", path)
	if err != nil {
		return err
	}
	defer conn.Close()
	_, err = conn.Exec("UPDATE items SET deleted_at=? WHERE id=?",
		time.Now().UTC().Format(time.RFC3339), id)
	return err
}

// TestGet_FoundAndMissing covers the read path: a live item is
// returned intact, while an unknown ID yields ErrNotFound.
func TestGet_FoundAndMissing(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "A retrievable fact",
		Body:  "Some body text.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	id, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)

	got, err := m.Get(ctx, id)
	testutil.AssertNoError(t, err)
	if got.ID != id {
		t.Fatalf("expected ID %q, got %q", id, got.ID)
	}
	if got.Title != item.Title {
		t.Fatalf("expected title %q, got %q", item.Title, got.Title)
	}

	if _, err := m.Get(ctx, "nonexistent-id"); !errors.Is(err, gomem.ErrNotFound) {
		t.Fatalf("expected ErrNotFound, got %v", err)
	}
}

// TestDelete_TombstonesAndIdempotent verifies Delete flips
// deleted_at, excludes the item from subsequent Gets, and
// reports ErrNotFound on repeat or unknown IDs.
func TestDelete_TombstonesAndIdempotent(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "A deletable fact",
		Body:  "Body.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	id, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)

	testutil.AssertNoError(t, m.Delete(ctx, id))

	if _, err := m.Get(ctx, id); !errors.Is(err, gomem.ErrNotFound) {
		t.Fatalf("expected ErrNotFound after delete, got %v", err)
	}
	if err := m.Delete(ctx, id); !errors.Is(err, gomem.ErrNotFound) {
		t.Fatalf("expected ErrNotFound on re-delete, got %v", err)
	}
	if err := m.Delete(ctx, "ghost"); !errors.Is(err, gomem.ErrNotFound) {
		t.Fatalf("expected ErrNotFound for ghost, got %v", err)
	}
}

// TestTouch_BumpsAccessAndMisses exercises the access counter:
// a live item is touched successfully, while an absent ID
// is missed.
func TestTouch_BumpsAccessAndMisses(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		Kind:  gomem.KindFact,
		Title: "A touchable fact",
		Body:  "Body.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	}
	id, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)

	testutil.AssertNoError(t, m.Touch(ctx, id))
	testutil.AssertNoError(t, m.Touch(ctx, id))

	got, err := m.Get(ctx, id)
	testutil.AssertNoError(t, err)
	if got.ID != id {
		t.Fatalf("expected item to remain accessible after touch, got %q", got.ID)
	}

	if err := m.Touch(ctx, "phantom"); !errors.Is(err, gomem.ErrNotFound) {
		t.Fatalf("expected ErrNotFound for phantom, got %v", err)
	}
}

// TestDouble_Close confirms idempotent Close.
func TestDouble_Close(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cfg := testConfig()
	cfg.Path = filepath.Join(dir, "double.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)

	testutil.AssertNoError(t, m.Close())
	testutil.AssertNoError(t, m.Close())
}

// TestConcurrent_ParallelRead_SequentialWrite fans out simultaneous
// reads (Recall) and writes (Remember) to exercise the store's
// concurrency contract: readers proceed in parallel (shared lock),
// writers are serialized (exclusive lock). Run with -race to catch
// data races in the locking.
func TestConcurrent_ParallelRead_SequentialWrite(t *testing.T) {
	ctx := context.Background()
	const (
		numReaders = 8
		numWriters = 6
	)
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	var (
		wg          sync.WaitGroup
		readErrors  = make([]error, numReaders)
		writeErrors = make([]error, numWriters)
	)

	// Readers: concurrent Recall calls (shared lock -> parallel).
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, readErrors[i] = m.Recall(ctx, gomem.RecallQuery{
				Goal: "concurrent recall reader",
				TopK: 5,
			})
		}(i)
	}

	// Writers: concurrent Remember calls (exclusive lock -> serialized).
	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, writeErrors[i] = m.Remember(ctx, gomem.Item{
				Kind:  gomem.KindFact,
				Title: fmt.Sprintf("concurrent writer item %d", i),
				Body:  fmt.Sprintf("distinct body for writer %d", i),
				Provenance: gomem.Provenance{
					Source:     "run",
					Confidence: gomem.Confidence(0.9),
				},
			})
		}(i)
	}

	wg.Wait()

	for i, err := range readErrors {
		if err != nil {
			t.Errorf("reader %d failed: %v", i, err)
		}
	}
	for i, err := range writeErrors {
		if err != nil {
			t.Errorf("writer %d failed: %v", i, err)
		}
	}

	// Every writer succeeded -> exactly numWriters live items.
	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems != numWriters {
		t.Fatalf("expected %d items after concurrent writes, got %d", numWriters, st.TotalItems)
	}
}

// TestCorrupt_DB_OnOpen covers detection, backup + recreate, and the
// ErrCorrupt classification.
func TestCorrupt_DB_OnOpen(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.db")

	cfg := testConfig()
	cfg.Path = path
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	valid, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	_, err = valid.Remember(ctx, gomem.Item{
		Kind:  gomem.KindFact,
		Title: "seeded before corruption",
		Body:  "some body",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	})
	testutil.AssertNoError(t, err)
	testutil.AssertNoError(t, valid.Close())

	data, err := os.ReadFile(path)
	testutil.AssertNoError(t, err)
	if len(data) < 64 {
		t.Fatalf("db too small to corrupt: %d bytes", len(data))
	}
	// Clobber the 16-byte SQLite header magic so the file is no
	// longer recognized as a database (caught on reopen).
	for i := 0; i < 16; i++ {
		data[i] ^= 0xFF
	}
	testutil.AssertNoError(t, os.WriteFile(path, data, 0o600))

	reopened, err := gomem.New(ctx, cfg)
	if err == nil {
		t.Fatal("expected ErrCorrupt on reopening a corrupt file")
	}
	if !errors.Is(err, gomem.ErrCorrupt) {
		t.Fatalf("expected ErrCorrupt, got %v", err)
	}
	if reopened == nil {
		t.Fatal("expected a usable (recreated) Memory alongside ErrCorrupt")
	}
	defer reopened.Close()

	st, err := reopened.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems != 0 {
		t.Fatalf("expected an empty recreated DB, got %d items", st.TotalItems)
	}
}

// TestOffline_Default_ZeroNet confirms Remember and Recall work with
// zero network under the default config (the dense leg degrades
// gracefully to the keyword leg).
func TestOffline_Default_ZeroNet(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cfg := testConfig()
	cfg.Path = filepath.Join(dir, "offline.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	_, err = m.Remember(ctx, gomem.Item{
		Kind:  gomem.KindFact,
		Title: "offline stored fact",
		Body:  "written without any network embedding service",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	})
	testutil.AssertNoError(t, err)

	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "offline stored fact network embedding",
		TopK: 3,
	})
	testutil.AssertNoError(t, err)
	if len(hits) == 0 {
		t.Fatal("expected recall to work offline")
	}
}

// TestDefaultConfig_DeadEmbedServerIsFast confirms the keyword-only
// fallback is cheap when nothing listens on the default llama.cpp
// embedder's port: only the first call pays (one short retry); later
// Remember/Recall calls skip the embedder during its cool-down instead of
// paying the 3.5 s backoff each time.
func TestDefaultConfig_DeadEmbedServerIsFast(t *testing.T) {
	ctx := context.Background()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	testutil.AssertNoError(t, err)
	dead := "http://" + ln.Addr().String()
	ln.Close()

	cfg := testConfig()
	cfg.Path = filepath.Join(t.TempDir(), "dead.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embed.APIURL = dead // default provider (llama.cpp) and default retries
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	start := time.Now()
	for i := 0; i < 2; i++ {
		_, err = m.Remember(ctx, gomem.Item{
			Kind:       gomem.KindFact,
			Title:      fmt.Sprintf("dead server fact %d", i),
			Body:       "stored while no embedding server is running",
			Provenance: gomem.Provenance{Source: "run", Confidence: gomem.Confidence(0.9)},
		})
		testutil.AssertNoError(t, err)
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "dead server fact", TopK: 3})
		testutil.AssertNoError(t, err)
		if len(hits) == 0 {
			t.Fatal("expected keyword-only recall to find the item")
		}
	}
	if d := time.Since(start); d > time.Second {
		t.Fatalf("2x Remember+Recall with a dead embedder took %v, want well under 3.5s", d)
	}
}

// TestStats covers counts by kind and sizes.
func TestStats(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()
	seedItems(t, ctx, m)

	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems != 4 {
		t.Fatalf("expected 4 items, got %d", st.TotalItems)
	}
	if st.ByKind[gomem.KindFact] != 2 {
		t.Fatalf("expected 2 facts, got %d", st.ByKind[gomem.KindFact])
	}
	if st.DBSizeBytes <= 0 {
		t.Fatalf("expected a positive DB size, got %d", st.DBSizeBytes)
	}
}

// TestReflect_Housekeeping covers salience recompute, eviction, and
// the report.
func TestReflect_Housekeeping(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cfg := testConfig()
	cfg.Path = filepath.Join(dir, "reflect.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	cfg.MaxItems = 3
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	for i := 0; i < 5; i++ {
		_, err := m.Remember(ctx, gomem.Item{
			Kind:  gomem.KindFact,
			Title: "reflect corpus item",
			Body:  "body reflecting housekeeping behaviour",
			Provenance: gomem.Provenance{
				Source:     "run",
				Confidence: gomem.Confidence(0.8),
			},
		})
		testutil.AssertNoError(t, err)
	}

	report, err := m.Reflect(ctx, gomem.TriggerPeriodic)
	testutil.AssertNoError(t, err)
	if report == nil {
		t.Fatal("expected a non-nil report")
	}
	if report.ItemsEvicted < 2 {
		t.Fatalf("expected >=2 evictions, got %+v", report)
	}

	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems > 3 {
		t.Fatalf("expected <=3 live items after eviction, got %d", st.TotalItems)
	}
}

// TestReflect_BudgetPressureVacuums confirms the vacuum runs, is reported
// by Stats, and leaves recall working.
func TestReflect_BudgetPressureVacuums(t *testing.T) {
	ctx := context.Background()
	cfg := testConfig()
	cfg.Path = filepath.Join(t.TempDir(), "vac.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	_, err = m.Remember(ctx, gomem.Item{ID: "a", Kind: gomem.KindFact, Title: "alpha topic",
		Provenance: gomem.Provenance{Source: "run", Confidence: gomem.Confidence(0.9)}})
	testutil.AssertNoError(t, err)
	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.LastVacuumAt != nil {
		t.Fatal("LastVacuumAt set before any vacuum")
	}
	_, err = m.Reflect(ctx, gomem.TriggerBudgetPressure)
	testutil.AssertNoError(t, err)
	st, err = m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.LastVacuumAt == nil {
		t.Fatal("LastVacuumAt not set after budget_pressure")
	}
	hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "alpha topic", TopK: 5})
	testutil.AssertNoError(t, err)
	if len(hits) != 1 || hits[0].ID != "a" {
		t.Fatalf("recall after vacuum: %+v", hits)
	}
}

// TestCapacity_EvictionLowestSalience covers the MaxItems overflow path.
func TestCapacity_EvictionLowestSalience(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	cfg := testConfig()
	cfg.Path = filepath.Join(dir, "capacity.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	cfg.MaxItems = 2
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	// Seed four items with distinct content, staggered so their creation
	// times (and thus recency/salience) differ. Capture each assigned ID.
	ids := make([]string, 4)
	for i := 0; i < 4; i++ {
		id, err := m.Remember(ctx, gomem.Item{
			Kind:  gomem.KindFact,
			Title: fmt.Sprintf("capacity marker variant %d", i),
			Body:  fmt.Sprintf("distinct eviction payload body %d exercising the ordering path", i),
			Provenance: gomem.Provenance{
				Source:     "run",
				Confidence: gomem.Confidence(0.7),
			},
		})
		testutil.AssertNoError(t, err)
		ids[i] = id
		time.Sleep(5 * time.Millisecond)
	}

	// Housekeeping pass enforces the capacity cap.
	if _, err := m.Reflect(ctx, gomem.TriggerPeriodic); err != nil {
		testutil.AssertNoError(t, err)
	}

	st, err := m.Stats(ctx)
	testutil.AssertNoError(t, err)
	if st.TotalItems > 2 {
		t.Fatalf("expected <=2 live items under the cap, got %d", st.TotalItems)
	}

	// Enumerate the survivors via a broad recall and confirm the eviction
	// dropped the LOWEST-salience (oldest) items: the newest must remain, the
	// oldest must be gone.
	hits, err := m.Recall(ctx, gomem.RecallQuery{
		Goal: "capacity marker variant distinct eviction payload body exercising the ordering path",
		TopK: 10,
	})
	if err != nil {
		t.Fatalf("recall: %v", err)
	}
	alive := map[string]bool{}
	for _, h := range hits {
		alive[h.ID] = true
	}
	if !alive[ids[3]] {
		t.Fatal("expected the newest item to survive eviction")
	}
	if alive[ids[0]] {
		t.Fatal("expected the oldest item to be evicted")
	}
}

func containsAny(s string, subs ...string) bool {
	lowered := toLowerASCII(s)
	for _, sub := range subs {
		if contains(lowered, toLowerASCII(sub)) {
			return true
		}
	}
	return false
}

func toLowerASCII(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

func contains(haystack, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		ok := true
		for j := 0; j < len(needle); j++ {
			if haystack[i+j] != needle[j] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// TestRemember_RevivesTombstonedID verifies that re-remembering a
// deleted ID (same content, then changed content) brings the item back.
func TestRemember_RevivesTombstonedID(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()

	item := gomem.Item{
		ID:    "revive-0001",
		Kind:  gomem.KindFact,
		Title: "Zebracorn migration notes",
		Body:  "Zebracorn pastures migrate seasonally.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.8),
		},
	}
	_, err := m.Remember(ctx, item)
	testutil.AssertNoError(t, err)

	for _, body := range []string{item.Body, "Zebracorn pastures no longer migrate."} {
		testutil.AssertNoError(t, m.Delete(ctx, item.ID))
		if _, err := m.Get(ctx, item.ID); !errors.Is(err, gomem.ErrNotFound) {
			t.Fatalf("expected ErrNotFound after delete, got %v", err)
		}

		item.Body = body
		_, err = m.Remember(ctx, item)
		testutil.AssertNoError(t, err)
		got, err := m.Get(ctx, item.ID)
		testutil.AssertNoError(t, err)
		if got.Body != body {
			t.Fatalf("revived body = %q, want %q", got.Body, body)
		}
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "zebracorn pastures", TopK: 3})
		testutil.AssertNoError(t, err)
		testutil.AssertContainsID(t, hits, item.ID)
	}
}

// gateEmbedder blocks texts containing "BLOCK" until release is closed and
// delegates everything else to a mock embedder.
type gateEmbedder struct {
	*testutil.MockEmbedder
	entered chan struct{}
	release chan struct{}
}

func newGateEmbedder() *gateEmbedder {
	return &gateEmbedder{
		MockEmbedder: testutil.NewMockEmbedder(testDim),
		entered:      make(chan struct{}, 8),
		release:      make(chan struct{}),
	}
}

func (g *gateEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	for _, tx := range texts {
		if strings.Contains(tx, "BLOCK") {
			g.entered <- struct{}{}
			<-g.release
			break
		}
	}
	return g.MockEmbedder.Embed(ctx, texts)
}

func TestRemember_SlowEmbedderDoesNotBlockRecall(t *testing.T) {
	ctx := context.Background()
	emb := newGateEmbedder()
	m, done := newTestMemory(t, emb)
	defer done()
	seedItems(t, ctx, m)

	rememberDone := make(chan error, 1)
	go func() {
		_, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "BLOCK slow item", Body: "body"})
		rememberDone <- err
	}()
	<-emb.entered

	recallDone := make(chan error, 1)
	go func() {
		_, err := m.Recall(ctx, gomem.RecallQuery{Goal: "composite index database", TopK: 3})
		recallDone <- err
	}()
	select {
	case err := <-recallDone:
		testutil.AssertNoError(t, err)
	case <-time.After(5 * time.Second):
		t.Fatal("Recall blocked behind a Remember that is still embedding")
	}
	select {
	case <-rememberDone:
		t.Fatal("Remember returned before the embedder was released")
	default:
	}
	close(emb.release)
	testutil.AssertNoError(t, <-rememberDone)
}

func TestRemember_CloseDuringEmbed(t *testing.T) {
	ctx := context.Background()
	emb := newGateEmbedder()
	m, _ := newTestMemory(t, emb)

	rememberDone := make(chan error, 1)
	go func() {
		_, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "BLOCK closing item", Body: "body"})
		rememberDone <- err
	}()
	<-emb.entered
	testutil.AssertNoError(t, m.Close())
	close(emb.release)
	if err := <-rememberDone; !errors.Is(err, gomem.ErrClosed) {
		t.Fatalf("expected ErrClosed, got %v", err)
	}
}

func TestRememberMany_Atomic(t *testing.T) {
	ctx := context.Background()
	good := gomem.Item{Kind: gomem.KindFact, Title: "valid item", Body: "body"}
	for name, bad := range map[string]struct {
		item gomem.Item
		want error
	}{
		"invalid":        {gomem.Item{Kind: gomem.KindFact}, gomem.ErrInvalid},
		"missing target": {gomem.Item{Kind: gomem.KindFact, Title: "t", Supersedes: "nope"}, gomem.ErrNotFound},
	} {
		t.Run(name, func(t *testing.T) {
			m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
			defer done()
			_, err := m.RememberMany(ctx, []gomem.Item{good, bad.item, good})
			if !errors.Is(err, bad.want) {
				t.Fatalf("expected %v, got %v", bad.want, err)
			}
			st, err := m.Stats(ctx)
			testutil.AssertNoError(t, err)
			if st.TotalItems != 0 {
				t.Fatalf("batch not all-or-nothing: %d items stored", st.TotalItems)
			}
		})
	}
}

func TestRememberMany_EquivalentToSequential(t *testing.T) {
	ctx := context.Background()
	items := testutil.NewFixture(1).Items(20)
	for i := range items {
		items[i].ID = fmt.Sprintf("many-%02d", i)
		items[i].Provenance.CreatedAt = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	}
	items[5].Supersedes = items[4].ID // link within the batch
	items[7].ID = items[6].ID         // repeated ID: last writer wins

	seq, doneSeq := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer doneSeq()
	bat, doneBat := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer doneBat()

	var wantIDs []string
	for _, it := range items {
		id, err := seq.Remember(ctx, it)
		testutil.AssertNoError(t, err)
		wantIDs = append(wantIDs, id)
	}
	gotIDs, err := bat.RememberMany(ctx, items)
	testutil.AssertNoError(t, err)
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("ids differ: %v vs %v", gotIDs, wantIDs)
	}
	for _, id := range wantIDs {
		a, err := seq.Get(ctx, id)
		testutil.AssertNoError(t, err)
		b, err := bat.Get(ctx, id)
		testutil.AssertNoError(t, err)
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("item %s differs:\n%+v\n%+v", id, a, b)
		}
	}
	q := gomem.RecallQuery{Goal: "optimize database query index", TopK: 5}
	a, err := seq.Recall(ctx, q)
	testutil.AssertNoError(t, err)
	b, err := bat.Recall(ctx, q)
	testutil.AssertNoError(t, err)
	// Scores drift with the wall clock; compare the ranking.
	if len(a) == 0 || len(a) != len(b) {
		t.Fatalf("recall sizes differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			t.Fatalf("recall rank %d differs: %s vs %s", i, a[i].ID, b[i].ID)
		}
	}
}

func benchRemember(b *testing.B, batch bool) {
	ctx := context.Background()
	cfg := testConfig()
	cfg.Path = filepath.Join(b.TempDir(), "bench.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	m, err := gomem.New(ctx, cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer m.Close()
	items := testutil.NewFixture(1).Items(1000)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := range items {
			items[i].ID = fmt.Sprintf("b%d-%d", n, i)
		}
		if batch {
			_, err = m.RememberMany(ctx, items)
			if err != nil {
				b.Fatal(err)
			}
			continue
		}
		for _, it := range items {
			if _, err := m.Remember(ctx, it); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func BenchmarkRememberMany_1K(b *testing.B)       { benchRemember(b, true) }
func BenchmarkRememberSequential_1K(b *testing.B) { benchRemember(b, false) }

// TestDenseIndexMemory_Lifecycle drives the in-memory dense index through
// the facade: recall after Remember, Delete, revival, an overwrite and
// Reflect, with readers running alongside a writer (run with -race).
func TestDenseIndexMemory_Lifecycle(t *testing.T) {
	ctx := context.Background()
	cfg := gomem.DefaultConfig()
	cfg.Path = filepath.Join(t.TempDir(), "idx.db")
	cfg.Logger = slog.New(slog.DiscardHandler)
	cfg.Embedder = testutil.NewMockEmbedder(testDim)
	cfg.DenseIndex = "memory"
	m, err := gomem.New(ctx, cfg)
	testutil.AssertNoError(t, err)
	defer m.Close()

	mk := func(id, title string) gomem.Item {
		return gomem.Item{ID: id, Kind: gomem.KindFact, Title: title,
			Provenance: gomem.Provenance{Source: "run", Confidence: gomem.Confidence(0.9)}}
	}
	_, err = m.RememberMany(ctx, []gomem.Item{mk("a", "alpha topic"), mk("b", "beta subject")})
	testutil.AssertNoError(t, err)
	recallIDs := func(goal string) []string {
		hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: goal, TopK: 10})
		testutil.AssertNoError(t, err)
		var ids []string
		for _, h := range hits {
			ids = append(ids, h.ID)
		}
		return ids
	}
	has := func(ids []string, id string) bool {
		for _, x := range ids {
			if x == id {
				return true
			}
		}
		return false
	}
	if !has(recallIDs("alpha topic"), "a") {
		t.Fatal("a not recalled")
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					if _, err := m.Recall(ctx, gomem.RecallQuery{Goal: "alpha topic", TopK: 5}); err != nil {
						t.Errorf("recall: %v", err)
						return
					}
				}
			}
		}()
	}
	for i := 0; i < 20; i++ {
		_, err := m.Remember(ctx, mk(fmt.Sprintf("n%d", i), fmt.Sprintf("noise %d", i)))
		testutil.AssertNoError(t, err)
	}
	testutil.AssertNoError(t, m.Delete(ctx, "a"))
	testutil.AssertNoError(t, m.Touch(ctx, "b"))
	close(stop)
	wg.Wait()

	if has(recallIDs("alpha topic"), "a") {
		t.Fatal("deleted item still recalled")
	}
	_, err = m.Remember(ctx, mk("a", "alpha topic")) // revive
	testutil.AssertNoError(t, err)
	if !has(recallIDs("alpha topic"), "a") {
		t.Fatal("revived item not recalled")
	}
	_, err = m.Remember(ctx, mk("b", "gamma replacement"))
	testutil.AssertNoError(t, err)
	if ids := recallIDs("gamma replacement"); len(ids) == 0 || ids[0] != "b" {
		t.Fatalf("overwritten item not on top: %v", ids)
	}
	_, err = m.Reflect(ctx, gomem.TriggerPeriodic)
	testutil.AssertNoError(t, err)
	if !has(recallIDs("alpha topic"), "a") {
		t.Fatal("a lost after Reflect")
	}
}

// TestRemember_TierUnspecifiedAndExplicit covers decision D1: a nil Tier
// starts a new item hot and keeps the stored tier on update, while an
// explicit value (including 0, cold) is stored and overwrites.
func TestRemember_TierUnspecifiedAndExplicit(t *testing.T) {
	ctx := context.Background()
	m, closeFn := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer closeFn()
	tierOf := func(id string) int {
		t.Helper()
		it, err := m.Get(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		if it.Tier == nil {
			t.Fatalf("Get(%s).Tier is nil", id)
		}
		return *it.Tier
	}

	id, err := m.Remember(ctx, gomem.Item{ID: "a", Kind: "fact", Title: "alpha"})
	if err != nil {
		t.Fatal(err)
	}
	if got := tierOf(id); got != 2 {
		t.Fatalf("new item with nil Tier: tier = %d, want 2 (hot)", got)
	}
	if _, err := m.Remember(ctx, gomem.Item{ID: "a", Kind: "fact", Title: "alpha", Tier: gomem.Tier(0)}); err != nil {
		t.Fatal(err)
	}
	if got := tierOf("a"); got != 0 {
		t.Fatalf("explicit cold: tier = %d, want 0", got)
	}
	// An update that leaves Tier nil must not reset the tier.
	if _, err := m.Remember(ctx, gomem.Item{ID: "a", Kind: "fact", Title: "alpha v2"}); err != nil {
		t.Fatal(err)
	}
	if got := tierOf("a"); got != 0 {
		t.Fatalf("update with nil Tier changed tier to %d, want 0 kept", got)
	}
	// Get then Remember round-trips the tier.
	it, _ := m.Get(ctx, "a")
	it.Title = "alpha v3"
	if _, err := m.Remember(ctx, it); err != nil {
		t.Fatal(err)
	}
	if got := tierOf("a"); got != 0 {
		t.Fatalf("round trip changed tier to %d", got)
	}
	if _, err := m.Remember(ctx, gomem.Item{ID: "b", Kind: "fact", Title: "beta", Tier: gomem.Tier(9)}); err != nil {
		t.Fatal(err)
	}
	if got := tierOf("b"); got != 2 {
		t.Fatalf("out-of-range tier clamped to %d, want 2", got)
	}
}

// TestRemember_ConfidenceUnspecifiedAndExplicitZero covers decision D2: nil
// Confidence is stored as 1, an explicit 0 is kept (and so excluded by a
// MinConfidence filter), out-of-range values are clamped and NaN is invalid.
func TestRemember_ConfidenceUnspecifiedAndExplicitZero(t *testing.T) {
	ctx := context.Background()
	m, done := newTestMemory(t, testutil.NewMockEmbedder(testDim))
	defer done()
	conf := func(id string) float64 {
		t.Helper()
		it, err := m.Get(ctx, id)
		testutil.AssertNoError(t, err)
		if it.Provenance.Confidence == nil {
			t.Fatalf("Get(%s).Provenance.Confidence is nil", id)
		}
		return *it.Provenance.Confidence
	}
	put := func(id string, c *float64) error {
		_, err := m.Remember(ctx, gomem.Item{ID: id, Kind: gomem.KindFact, Title: "shared goal " + id,
			Provenance: gomem.Provenance{Confidence: c}})
		return err
	}
	testutil.AssertNoError(t, put("unset", nil))
	testutil.AssertNoError(t, put("zero", gomem.Confidence(0)))
	testutil.AssertNoError(t, put("high", gomem.Confidence(7)))
	for id, want := range map[string]float64{"unset": 1, "zero": 0, "high": 1} {
		if got := conf(id); got != want {
			t.Errorf("%s: confidence = %v, want %v", id, got, want)
		}
	}
	if err := put("nan", gomem.Confidence(math.NaN())); !errors.Is(err, gomem.ErrInvalid) {
		t.Fatalf("NaN confidence: err = %v, want ErrInvalid", err)
	}
	hits, err := m.Recall(ctx, gomem.RecallQuery{Goal: "shared goal", MinConfidence: 0.5, TopK: 10})
	testutil.AssertNoError(t, err)
	for _, h := range hits {
		if h.ID == "zero" {
			t.Fatal("untrusted (confidence 0) item passed MinConfidence 0.5")
		}
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want the 2 trusted items", len(hits))
	}
}
