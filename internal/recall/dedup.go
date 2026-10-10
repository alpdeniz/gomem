package recall

import "github.com/alpdeniz/gomem/internal/textutil"

// dedupNear removes near-duplicate candidates (those whose token
// Jaccard similarity with an already-kept, higher-scored candidate
// meets the threshold). Order is preserved.
func dedupNear(cands []Candidate, threshold float64) []Candidate {
	if threshold <= 0 {
		return cands
	}
	keptSets := make([]map[string]struct{}, 0, len(cands))
	out := make([]Candidate, 0, len(cands))
	for _, c := range cands {
		set := textutil.TokenSet(c.Title + " " + c.Body)
		dup := false
		for _, kept := range keptSets {
			if textutil.Jaccard(set, kept) >= threshold {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, c)
			keptSets = append(keptSets, set)
		}
	}
	return out
}
