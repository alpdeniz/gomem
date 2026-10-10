package recall

import (
	"math"
	"testing"
)

func TestMinMax(t *testing.T) {
	got := minMax(map[string]float64{"lo": 2, "mid": 3, "hi": 4})
	if got["lo"] != 0 || got["mid"] != 0.5 || got["hi"] != 1 {
		t.Fatalf("got %v", got)
	}
	if got := minMax(map[string]float64{"only": 7}); got["only"] != 0.5 {
		t.Fatalf("a single score should be neutral, got %v", got)
	}
	if got := minMax(nil); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
}

func TestRRF(t *testing.T) {
	got := rrf(map[string]float64{"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1})
	want := map[string]float64{"a": 1, "b": 61.0 / 62, "c": 61.0 / 63, "d": 61.0 / 64} // ties by ID
	for id, w := range want {
		if math.Abs(got[id]-w) > 1e-12 {
			t.Fatalf("%s: got %v, want %v", id, got[id], w)
		}
	}
}

// TestNormalizeLegs checks each mode's mapping, including that BM25 is
// negated (more negative is more relevant) in every mode.
func TestNormalizeLegs(t *testing.T) {
	dense := map[string]float64{"x": 1, "y": 0, "z": -1}
	sparse := map[string]float64{"x": -8, "y": -2, "z": -1}
	for _, tc := range []struct {
		mode string
		dn   map[string]float64
	}{
		{"", map[string]float64{"x": 1, "y": 0.5, "z": 0}},
		{"legacy", map[string]float64{"x": 1, "y": 0.5, "z": 0}},
		{"minmax", map[string]float64{"x": 1, "y": 0.5, "z": 0}},
		{"rrf", map[string]float64{"x": 1, "y": 61.0 / 62, "z": 61.0 / 63}},
	} {
		e := &Engine{cfg: EngineConfig{FusionMode: tc.mode}}
		dn, sn := e.normalizeLegs(dense, sparse)
		for id, w := range tc.dn {
			if math.Abs(dn[id]-w) > 1e-12 {
				t.Fatalf("%q dense %s: got %v, want %v", tc.mode, id, dn[id], w)
			}
		}
		if !(sn["x"] > sn["y"] && sn["y"] > sn["z"]) || sn["x"] != 1 {
			t.Fatalf("%q sparse not ordered by relevance: %v", tc.mode, sn)
		}
	}
}
