// Package testutil provides deterministic test doubles and data
// generators: a hash-based mock embedder, item fixtures, and small
// assertion helpers.
package testutil

import (
	"context"

	"github.com/alpdeniz/gomem/internal/embed"
)

// MockEmbedder is a deterministic, zero-network embedder for tests.
// By default it hashes each text into a dim-dimensional vector (so
// similar texts land nearby); set Fixed to force every input to the
// same vector (useful for isolating behaviour).
type MockEmbedder struct {
	dim   int
	fixed []float32
}

// NewMockEmbedder constructs a mock embedder at the given dimension.
func NewMockEmbedder(dim int) *MockEmbedder {
	if dim <= 0 {
		dim = 256
	}
	return &MockEmbedder{dim: dim}
}

// WithFixed returns a copy that maps every input to the given vector.
func (m *MockEmbedder) WithFixed(v []float32) *MockEmbedder {
	c := *m
	c.fixed = v
	return &c
}

// Dim reports the vector dimensionality.
func (m *MockEmbedder) Dim() int { return m.dim }

// Embed returns one deterministic vector per input.
func (m *MockEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		if m.fixed != nil {
			out[i] = m.fixed
		} else {
			out[i] = embed.HashVector(t, m.dim)
		}
	}
	return out, nil
}

// ErrEmbedder is a stub embedder that always fails, used to exercise
// graceful degradation (dense leg skipped, FTS-only results).
type ErrEmbedder struct{ dim int }

// NewErrEmbedder constructs a always-failing embedder.
func NewErrEmbedder(dim int) *ErrEmbedder {
	if dim <= 0 {
		dim = 256
	}
	return &ErrEmbedder{dim: dim}
}

// Dim reports the dimensionality.
func (e *ErrEmbedder) Dim() int { return e.dim }

// Embed always returns an error.
func (e *ErrEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return nil, context.DeadlineExceeded
}
