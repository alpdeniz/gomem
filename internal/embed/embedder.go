// Package embed defines the pluggable Embedder seam and its
// implementations: a zero-network offline embedder (feature hashing),
// a llama.cpp server adapter, an optional OpenAI-compatible HTTP
// adapter, and a content-hash embedding cache.
package embed

import (
	"context"
	"encoding/binary"
	"hash/fnv"
	"math"
	"strings"
	"unicode"

	"github.com/alpdeniz/gomem/internal/textutil"
)

// Embedder turns text into fixed-size vectors.
type Embedder interface {
	// Embed returns one vector per input text, in order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dim is the produced vector dimensionality.
	Dim() int
}

// HashVector deterministically projects a text into a dim-dimensional
// L2-normalized vector using the feature-hashing trick over word
// tokens and character trigrams. Texts sharing tokens/n-grams land
// closer in cosine space. It is the shared core behind the offline
// embedder and the test mock.
func HashVector(text string, dim int) []float32 {
	if dim <= 0 {
		dim = 256
	}
	v := make([]float32, dim)
	for _, f := range features(text) {
		h := fnvHash(f)
		idx := int(h % uint64(dim))
		sign := float32(1)
		if h&(1<<32) != 0 {
			sign = -1
		}
		v[idx] += sign
	}
	return l2Normalize(v)
}

// features extracts word tokens and character trigrams from text.
func features(text string) []string {
	low := strings.ToLower(text)
	out := textutil.Words(low)
	cleaned := make([]rune, 0, len(low))
	for _, r := range low {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cleaned = append(cleaned, r)
		}
	}
	if len(cleaned) >= 3 {
		tris := make([]string, 0, len(cleaned)-2)
		for i := 0; i+3 <= len(cleaned); i++ {
			tris = append(tris, string(cleaned[i:i+3]))
		}
		out = append(out, tris...)
	}
	return out
}

func fnvHash(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

func l2Normalize(v []float32) []float32 {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		s := float32(norm)
		for i := range v {
			v[i] /= s
		}
	}
	return v
}

// EncodeVec serializes a float32 slice to a little-endian BLOB.
func EncodeVec(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// DecodeVec reverses EncodeVec.
func DecodeVec(b []byte) []float32 {
	n := len(b) / 4
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
