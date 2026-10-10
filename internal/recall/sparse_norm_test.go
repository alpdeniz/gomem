package recall

import (
	"math"
	"testing"
)

// TestNormalizeSparse_BoundsAndMonotonic proves the sparse leg is
// normalized onto [0,1] and that more-relevant (more negative FTS5 BM25)
// candidates score higher — the commensurability guarantee the dense
// leg already had. It also pins the full dynamic range: the best match maps
// to 1.0 and the weakest to 0.0.
func TestNormalizeSparse_BoundsAndMonotonic(t *testing.T) {
	raw := map[string]float64{
		"best":  -8.0,
		"good":  -4.0,
		"mid":   -2.0,
		"poor":  -0.5,
		"worst": -0.1,
	}
	got := normalizeSparse(raw)

	for id, v := range got {
		if v < 0 || v > 1 {
			t.Fatalf("%s: %v out of [0,1]", id, v)
		}
	}
	if math.Abs(got["best"]-1) > 1e-9 {
		t.Fatalf("best should map to 1.0, got %v", got["best"])
	}
	if math.Abs(got["worst"]) > 1e-9 {
		t.Fatalf("worst should map to 0.0, got %v", got["worst"])
	}

	order := []string{"worst", "poor", "mid", "good", "best"}
	for i := 1; i < len(order); i++ {
		if got[order[i]] <= got[order[i-1]] {
			t.Fatalf("not strictly increasing at %s: %v <= %v",
				order[i], got[order[i]], got[order[i-1]])
		}
	}
}

// TestNormalizeSparse_MidMatchIsMidway guards the reviewer's concern that
// the keyword leg "dominates the composite more than beta suggests": a
// mid-relevance sparse match must land in the middle of [0,1], not be pinned
// to an extreme that would let it swamp the other legs.
func TestNormalizeSparse_MidMatchIsMidway(t *testing.T) {
	raw := map[string]float64{
		"best":  -8.0,
		"mid":   -4.0,
		"worst": 0.0,
	}
	got := normalizeSparse(raw)
	// "mid" is halfway between best and worst in raw relevance, so its
	// normalized score must be ~0.5 — squarely between the extremes.
	if math.Abs(got["mid"]-0.5) > 1e-9 {
		t.Fatalf("mid-relevance match should normalize to ~0.5, got %v", got["mid"])
	}
}

// TestNormalizeSparse_Degenerate confirms an all-tied batch maps to a
// neutral 0.5 rather than dividing by zero or collapsing to an extreme.
func TestNormalizeSparse_Degenerate(t *testing.T) {
	got := normalizeSparse(map[string]float64{"a": -3, "b": -3, "c": -3})
	for id, v := range got {
		if math.Abs(v-0.5) > 1e-9 {
			t.Fatalf("%s: expected 0.5, got %v", id, v)
		}
	}
}

// TestNormalizeSparse_Empty covers the no-sparse-hits case.
func TestNormalizeSparse_Empty(t *testing.T) {
	if got := normalizeSparse(nil); len(got) != 0 {
		t.Fatalf("expected empty, got %v", got)
	}
}

// TestBuildFTSQuery_ORJoin pins the breadth-recall contract: multiple
// tokens are joined with OR so the sparse leg surfaces documents matching
// ANY query term, while BM25 ordering still ranks multi-term matches higher.
func TestBuildFTSQuery_ORJoin(t *testing.T) {
	got := buildFTSQuery("cache invalidation strategy")
	want := `"cache" OR "invalidation" OR "strategy"`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

// TestBuildFTSQuery_SingleToken confirms a one-word query produces a
// bare quoted token with no dangling OR.
func TestBuildFTSQuery_SingleToken(t *testing.T) {
	if got := buildFTSQuery("hello"); got != `"hello"` {
		t.Fatalf("got %q", got)
	}
}

// TestBuildFTSQuery_Empty covers non-alphanumeric-only input.
func TestBuildFTSQuery_Empty(t *testing.T) {
	if got := buildFTSQuery("!!! ???"); got != "" {
		t.Fatalf("got %q", got)
	}
}

// TestNormalizeSparse_OldTransformWasBroken documents why the naive
// 1/(1+bm25) was insufficient: for realistic negative FTS5 BM25 it left
// [0,1] (negative for strong matches, >1 for weak, div-by-zero at -1), so the
// composite was no longer a true convex combination. Our transform keeps
// the same inputs in [0,1].
func TestNormalizeSparse_OldTransformWasBroken(t *testing.T) {
	cases := []struct {
		name string
		bm25 float64
	}{
		{"strong_match", -8.0},
		{"weak_match", -0.3},
		{"divide_by_zero", -1.0},
	}
	for _, tc := range cases {
		old := 1 / (1 + tc.bm25)
		outOfRange := old < 0 || old > 1 || math.IsInf(old, 0)
		if !outOfRange {
			t.Fatalf("%s (bm25=%v): expected the old 1/(1+bm25) to leave [0,1], got %v",
				tc.name, tc.bm25, old)
		}
		fixed := normalizeSparse(map[string]float64{"x": tc.bm25})
		if v := fixed["x"]; v < 0 || v > 1 {
			t.Fatalf("%s: fixed transform %v out of [0,1]", tc.name, v)
		}
	}
}

// TestBuildFTSQuery_Unicode confirms letters and digits of any script
// become tokens.
func TestBuildFTSQuery_Unicode(t *testing.T) {
	got := buildFTSQuery("veritabanı, индекс! 索引 café2")
	want := `"veritabanı" OR "индекс" OR "索引" OR "café2"`
	if got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}
