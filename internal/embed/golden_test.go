package embed

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// TestHashVector_ASCIIGolden pins the exact vectors the offline embedder
// produced for ASCII text before Unicode support, so databases embedded
// with it keep working. Each value is the SHA-256 of EncodeVec(vector).
func TestHashVector_ASCIIGolden(t *testing.T) {
	golden := map[string]string{
		"Composite indexes accelerate filtered database queries, v2!": "9cede5011363fcf517d5a5fc51bc36da50c19fe6ce9e2fe19388baff0839c18d",
		"Hello, World! foo_bar-baz 42":                                "5b5c1f97569081e66b9df5f440e33ac646ed2abbf94c20a40b424a1b16a5c210",
		"Hi":                                                          "3622f88f2629fbcb76092493fe03c9d8c198c20aa3397e7cfb75adb84a391483",
	}
	for text, want := range golden {
		sum := sha256.Sum256(EncodeVec(HashVector(text, 256)))
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("HashVector(%q) changed: got %s, want %s", text, got, want)
		}
	}
}

// TestHashVector_NonASCII confirms non-ASCII letters now contribute
// features instead of vanishing.
func TestHashVector_NonASCII(t *testing.T) {
	a := HashVector("veritabanı indeksi", 256)
	b := HashVector("データベース索引", 256)
	if cosine(a, b) > 0.5 {
		t.Fatalf("unrelated non-ASCII texts should differ, cosine %v", cosine(a, b))
	}
	if cosine(a, HashVector("veritabanı indeksi", 256)) < 0.999 {
		t.Fatal("identical text should match itself")
	}
	var norm float64
	for _, x := range b {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		t.Fatal("Japanese text produced a zero vector")
	}
}
