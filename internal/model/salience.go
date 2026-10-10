package model

import (
	"math"
	"time"
)

// Params holds the tunables for salience computation.
type Params struct {
	HalfLifeDays float64
	FreqCap      int
	WConfidence  float64
	WFrequency   float64
	WRecency     float64
}

// Recency returns exp(-ln2 * age_days / half_life) clamped to [0,1].
func Recency(now, ref time.Time, halfLifeDays float64) float64 {
	if halfLifeDays <= 0 {
		halfLifeDays = 30
	}
	ageDays := max(now.Sub(ref), 0).Hours() / 24
	return clamp01(math.Exp(-math.Ln2 * ageDays / halfLifeDays))
}

// Reinforce returns log1p(count)/log1p(cap) clamped to [0,1].
func Reinforce(count, cap int) float64 {
	if cap <= 0 {
		cap = 100
	}
	return clamp01(math.Log1p(float64(max(count, 0))) / math.Log1p(float64(cap)))
}

// Scalar computes the unified salience scalar in [0,1]:
// clamp01(w_c*confidence + w_f*reinforce + w_r*recency).
func Scalar(confidence float64, count int, now, ref time.Time, p Params) float64 {
	return clamp01(p.WConfidence*clamp01(confidence) +
		p.WFrequency*Reinforce(count, p.FreqCap) +
		p.WRecency*Recency(now, ref, p.HalfLifeDays))
}

// Of is the salience of an item with the given usage at time now.
func (p Params) Of(u Usage, now time.Time) float64 {
	return Scalar(u.Confidence, u.AccessCount, now, u.Ref(), p)
}

func clamp01(x float64) float64 { return min(max(x, 0), 1) }
