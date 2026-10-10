package textutil

import (
	"reflect"
	"testing"
)

func TestCleanTags(t *testing.T) {
	got := CleanTags([]string{" DB ", "db", "", "Cache", "  "})
	want := []string{"db", "cache"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestTokenSetJaccard_Unicode(t *testing.T) {
	a := TokenSet("Veritabanı İNDEKS 索引 v2")
	for _, w := range []string{"veritabanı", "索引", "v2"} {
		if _, ok := a[w]; !ok {
			t.Fatalf("missing token %q in %v", w, a)
		}
	}
	if j := Jaccard(TokenSet("индекс базы"), TokenSet("Индекс Базы!")); j != 1 {
		t.Fatalf("jaccard = %v, want 1", j)
	}
}
