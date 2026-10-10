package embed

import (
	"context"
	"math"
	"testing"
)

func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.NaN()
	}
	var dot, na, nb float64
	for i := range a {
		af, bf := float64(a[i]), float64(b[i])
		dot += af * bf
		na += af * af
		nb += bf * bf
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestHashVector_Deterministic(t *testing.T) {
	a := HashVector("the quick brown fox jumps over the lazy dog", 256)
	b := HashVector("the quick brown fox jumps over the lazy dog", 256)
	if len(a) != 256 {
		t.Fatalf("expected dim 256, got %d", len(a))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-deterministic at index %d", i)
		}
	}
}

func TestHashVector_Normalized(t *testing.T) {
	v := HashVector("some text to normalize properly", 128)
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if math.Abs(norm-1) > 1e-3 {
		t.Fatalf("expected unit norm, got %.4f", norm)
	}
}

func TestHashVector_SimilarityOrdering(t *testing.T) {
	dim := 256
	query := HashVector("optimize database query performance with indexes", dim)
	related := HashVector("improve database query speed using composite indexes", dim)
	unrelated := HashVector("recipe for chocolate cake with vanilla frosting", dim)

	simRelated := cosine(query, related)
	simUnrelated := cosine(query, unrelated)
	if !(simRelated > simUnrelated) {
		t.Fatalf("expected related (%.3f) to beat unrelated (%.3f)", simRelated, simUnrelated)
	}
}

func TestOfflineEmbedder_Embed(t *testing.T) {
	o := NewOfflineEmbedder(64)
	if o.Dim() != 64 {
		t.Fatalf("expected dim 64, got %d", o.Dim())
	}
	vecs, err := o.Embed(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 64 {
			t.Fatalf("vector[%d] dim %d, want 64", i, len(v))
		}
	}
}

func TestEncodeDecodeVec_RoundTrip(t *testing.T) {
	in := []float32{1.5, -2.25, 0, 3.75}
	blob := EncodeVec(in)
	if len(blob) != len(in)*4 {
		t.Fatalf("expected %d bytes, got %d", len(in)*4, len(blob))
	}
	out := DecodeVec(blob)
	if len(out) != len(in) {
		t.Fatalf("decoded len %d, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Fatalf("round-trip mismatch at %d: %v != %v", i, out[i], in[i])
		}
	}
}
