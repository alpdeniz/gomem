package testutil

import (
	"testing"

	"github.com/alpdeniz/gomem"
)

// AssertNoError fails the test if err is non-nil.
func AssertNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// AssertError fails the test if err is nil.
func AssertError(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
}

// AssertTrue fails the test if cond is false.
func AssertTrue(t *testing.T, cond bool, msg string) {
	t.Helper()
	if !cond {
		t.Fatalf("condition failed: %s", msg)
	}
}

// AssertContainsID fails unless the hits include the wanted ID.
func AssertContainsID(t *testing.T, hits []gomem.Hit, id string) {
	t.Helper()
	for _, h := range hits {
		if h.ID == id {
			return
		}
	}
	t.Fatalf("hits do not contain id %q (got %d hits)", id, len(hits))
}

// AssertSortedDesc fails unless the hit scores are non-increasing.
func AssertSortedDesc(t *testing.T, hits []gomem.Hit) {
	t.Helper()
	for i := 1; i < len(hits); i++ {
		if hits[i].Score > hits[i-1].Score {
			t.Fatalf("scores not sorted descending at index %d (%v > %v)",
				i, hits[i].Score, hits[i-1].Score)
		}
	}
}

// AssertScoreRange fails unless every hit score lies in [lo, hi].
func AssertScoreRange(t *testing.T, hits []gomem.Hit, lo, hi float64) {
	t.Helper()
	for _, h := range hits {
		if h.Score < lo || h.Score > hi {
			t.Fatalf("score %v out of range [%v, %v] for id %s", h.Score, lo, hi, h.ID)
		}
	}
}
