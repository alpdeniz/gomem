package model

import (
	"math"
	"testing"
	"time"
)

func approx(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestRecency_Decay(t *testing.T) {
	now := time.Date(2026, 10, 4, 0, 0, 0, 0, time.UTC)
	halfLife := 30.0

	cases := []struct {
		ageDays float64
		want    float64
	}{
		{0, 1.0},
		{30, 0.5},
		{60, 0.25},
		{90, 0.125},
	}
	for _, tc := range cases {
		ref := now.AddDate(0, 0, -int(tc.ageDays))
		got := Recency(now, ref, halfLife)
		if !approx(got, tc.want, 1e-6) {
			t.Fatalf("age %.0fd: got %.4f, want %.4f", tc.ageDays, got, tc.want)
		}
	}
}

func TestReinforce_Scaling(t *testing.T) {
	cap := 100
	cases := []struct {
		count int
		want  float64
	}{
		{0, 0},
		{100, 1.0},
	}
	for _, tc := range cases {
		got := Reinforce(tc.count, cap)
		if !approx(got, tc.want, 1e-6) {
			t.Fatalf("count %d: got %.4f, want %.4f", tc.count, got, tc.want)
		}
	}
	// Monotonic in count.
	if !(Reinforce(10, cap) > Reinforce(1, cap)) {
		t.Fatal("expected reinforce to grow with count")
	}
}

func TestScalar_BoundsAndBlend(t *testing.T) {
	now := time.Date(2026, 10, 4, 0, 0, 0, 0, time.UTC)
	p := Params{HalfLifeDays: 30, FreqCap: 100, WConfidence: 0.2, WFrequency: 0.3, WRecency: 0.5}

	// Fresh, high-confidence, frequently accessed => high salience.
	fresh := now
	high := Scalar(1.0, 100, now, fresh, p)
	if high < 0.9 {
		t.Fatalf("expected high salience for fresh/confident/frequent, got %.3f", high)
	}

	// Stale, low-confidence, never accessed => low salience.
	stale := now.AddDate(0, 0, -120)
	low := Scalar(0.1, 0, now, stale, p)
	if low > 0.3 {
		t.Fatalf("expected low salience for stale/unconfident/cold, got %.3f", low)
	}

	// Ordering: high > low.
	if !(high > low) {
		t.Fatal("expected high > low salience")
	}
}
