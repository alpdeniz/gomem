// Package textutil holds small text helpers shared by the remember,
// recall and reflect packages so their rules cannot drift apart.
package textutil

import (
	"regexp"
	"strings"
)

var wordRe = regexp.MustCompile(`[\p{L}\p{N}]+`)

// Words returns the maximal runs of Unicode letters and digits in text.
func Words(text string) []string {
	return wordRe.FindAllString(text, -1)
}

// stopwords are common English function words that carry no topical signal
// and match nearly every document in natural-language text. The set is
// deliberately compact and leaves out short but meaningful tokens ("go",
// "c", "r", "up", "out", "no", "not", ...).
var stopwords = func() map[string]struct{} {
	set := map[string]struct{}{}
	for _, w := range strings.Fields(`a an the and or but nor if so than then
		of to in on at by for from with without into onto over about as
		is are was were be been being am do does did done doing have has had having
		i me my we our us you your he she it its they them their this that these those
		what which who whom whose when where why how
		can could should would will shall may might must
		there here very just also too any some such each both`) {
		set[w] = struct{}{}
	}
	return set
}()

// IsStopword reports whether the token is a common English stopword
// (case-insensitive).
func IsStopword(tok string) bool {
	_, ok := stopwords[strings.ToLower(tok)]
	return ok
}

// ContentWords returns the Words of text without stopwords. When every word
// is a stopword it returns all of them, so a query such as "the" still has
// something to match.
func ContentWords(text string) []string {
	words := Words(text)
	out := make([]string, 0, len(words))
	for _, w := range words {
		if !IsStopword(w) {
			out = append(out, w)
		}
	}
	if len(out) == 0 {
		return words
	}
	return out
}

// TokenSet builds a set of lowercased word tokens.
func TokenSet(text string) map[string]struct{} {
	set := map[string]struct{}{}
	for _, t := range Words(strings.ToLower(text)) {
		set[t] = struct{}{}
	}
	return set
}

// Jaccard returns the Jaccard similarity of two token sets in [0,1].
func Jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	smaller, larger := a, b
	if len(smaller) > len(larger) {
		smaller, larger = larger, smaller
	}
	inter := 0
	for tok := range smaller {
		if _, ok := larger[tok]; ok {
			inter++
		}
	}
	union := len(a) + len(b) - inter
	if union == 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// CleanTags lowercases, trims and de-duplicates tags, dropping empty ones.
func CleanTags(tags []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(tags))
	for _, t := range tags {
		t = strings.TrimSpace(strings.ToLower(t))
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	return out
}
