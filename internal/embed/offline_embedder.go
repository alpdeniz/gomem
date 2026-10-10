package embed

import "context"

// OfflineEmbedder is a tiny, deterministic, zero-network local
// vectorizer built on feature hashing. It lets the core library
// work fully offline (no API, no daemon).
type OfflineEmbedder struct {
	dim int
}

// NewOfflineEmbedder constructs the offline embedder at the given
// dimension (<=0 becomes 256).
func NewOfflineEmbedder(dim int) *OfflineEmbedder {
	if dim <= 0 {
		dim = 256
	}
	return &OfflineEmbedder{dim: dim}
}

// Dim implements Embedder.
func (o *OfflineEmbedder) Dim() int { return o.dim }

// Embed implements Embedder, projecting each text locally.
func (o *OfflineEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = HashVector(t, o.dim)
	}
	return out, nil
}
