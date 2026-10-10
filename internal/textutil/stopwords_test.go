package textutil

import (
	"reflect"
	"testing"
)

func TestIsStopword(t *testing.T) {
	for _, w := range []string{"the", "The", "how", "I", "is", "of"} {
		if !IsStopword(w) {
			t.Errorf("%q should be a stopword", w)
		}
	}
	for _, w := range []string{"go", "c", "r", "E0502", "40001", "database", "up", "not", "no", "x509"} {
		if IsStopword(w) {
			t.Errorf("%q must not be a stopword", w)
		}
	}
}

func TestContentWords(t *testing.T) {
	got := ContentWords("How do I fix the slow database in Go?")
	want := []string{"fix", "slow", "database", "Go"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	// All stopwords: fall back to every word.
	got = ContentWords("To be, or to be")
	if want := []string{"To", "be", "or", "to", "be"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if got := ContentWords("!!!"); len(got) != 0 {
		t.Fatalf("got %v", got)
	}
}
