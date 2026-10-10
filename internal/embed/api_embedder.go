package embed

import (
	"context"
	"encoding/json"
	"time"
)

// APIConfig configures the OpenAI-compatible HTTP embedder.
type APIConfig struct {
	APIURL     string
	APIKey     string
	Model      string
	Dim        int
	Timeout    time.Duration
	MaxRetries int
	BatchSize  int
	// SendDimensions controls the "dimensions" request field. nil sends it
	// (the default); point at false for servers that reject unknown fields.
	SendDimensions *bool
}

// APIEmbedder is the OPTIONAL network embedder speaking the
// OpenAI-compatible /embeddings contract (works with Qwen/DashScope,
// OpenAI, or any compatible endpoint). Like the llama.cpp embedder it is
// skipped for a short cool-down after a connection-level failure or 5xx.
type APIEmbedder struct {
	remote
	cfg APIConfig
}

// NewAPIEmbedder constructs the HTTP embedder.
func NewAPIEmbedder(cfg APIConfig) *APIEmbedder {
	cfg.Dim = positive(cfg.Dim, 512)
	cfg.Timeout = positive(cfg.Timeout, 10*time.Second)
	return &APIEmbedder{
		remote: newRemote(cfg.APIURL, cfg.APIKey, cfg.Timeout, cfg.MaxRetries, cfg.BatchSize, "gomem: embed"),
		cfg:    cfg,
	}
}

// Dim implements Embedder.
func (a *APIEmbedder) Dim() int { return a.cfg.Dim }

type apiRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type apiResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed implements Embedder, batching inputs and retrying transient failures.
func (a *APIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	return a.embed(ctx, texts, func(batch []string) any {
		req := apiRequest{Model: a.cfg.Model, Input: batch}
		if a.cfg.SendDimensions == nil || *a.cfg.SendDimensions {
			req.Dimensions = a.cfg.Dim
		}
		return req
	}, parseAPI)
}

func parseAPI(raw []byte, _ int) ([][]float32, error) {
	var ar apiResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return nil, err
	}
	out := make([][]float32, len(ar.Data))
	for _, d := range ar.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = d.Embedding
		}
	}
	return out, nil
}
