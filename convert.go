package gomem

import (
	"encoding/json"
	"fmt"
	"log/slog"

	"github.com/alpdeniz/gomem/internal/embed"
	"github.com/alpdeniz/gomem/internal/model"
	"github.com/alpdeniz/gomem/internal/remember"
)

// toInput normalizes the public Item for the persistence layer.
func (it Item) toInput(maxPayloadKB int) (remember.Input, error) {
	payload, err := marshalPayload(it.Payload, maxPayloadKB)
	if err != nil {
		return remember.Input{}, err
	}
	conf := 1.0 // unspecified; the value is copied, never aliased to the caller's
	if it.Provenance.Confidence != nil {
		conf = *it.Provenance.Confidence
	}
	return remember.Input{
		ID:      it.ID,
		Kind:    string(it.Kind),
		Title:   it.Title,
		Body:    it.Body,
		Payload: payload,
		Provenance: model.ProvDoc{
			Source:       it.Provenance.Source,
			RunID:        it.Provenance.RunID,
			NodeRole:     it.Provenance.NodeRole,
			Confidence:   conf,
			EvidenceRefs: it.Provenance.EvidenceRefs,
			CreatedAt:    it.Provenance.CreatedAt,
		},
		Tier:       it.Tier,
		Tags:       it.Tags,
		Supersedes: it.Supersedes,
	}, nil
}

func marshalPayload(payload any, maxKB int) ([]byte, error) {
	if payload == nil {
		return nil, nil
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalid, err)
	}
	if maxKB > 0 && len(b) > maxKB*1024 {
		return nil, fmt.Errorf("%w: payload exceeds %dKB", ErrInvalid, maxKB)
	}
	return b, nil
}

// publicProvenance is the output form of a stored provenance document: its
// Confidence is always non-nil.
func publicProvenance(pd model.ProvDoc) Provenance {
	return Provenance{
		Source:       pd.Source,
		RunID:        pd.RunID,
		NodeRole:     pd.NodeRole,
		Confidence:   &pd.Confidence,
		EvidenceRefs: pd.EvidenceRefs,
		CreatedAt:    pd.CreatedAt,
	}
}

func kindsToStrings(kinds []Kind) []string {
	if len(kinds) == 0 {
		return nil
	}
	out := make([]string, len(kinds))
	for i, k := range kinds {
		out[i] = string(k)
	}
	return out
}

// cacheNamespace scopes cached vectors to the embedder configuration so
// switching provider, model or dimension never returns stale vectors.
func cacheNamespace(cfg Config, emb embed.Embedder) string {
	if cfg.Embedder != nil {
		return fmt.Sprintf("%T|%d", emb, emb.Dim())
	}
	return fmt.Sprintf("%s|%s|%d", cfg.Embed.Provider, cfg.Embed.Model, cfg.Dim)
}

// buildEmbedder selects the embedder implementation from a validated
// config (Embed.Provider is canonical). The DEFAULT is the llama.cpp
// server so recall is genuinely semantic out of the box; "offline" opts
// into the zero-network feature-hash embedder, and "openai" uses the
// OpenAI-compatible HTTP adapter.
func buildEmbedder(cfg Config, lg *slog.Logger) embed.Embedder {
	if cfg.Embedder != nil {
		return cfg.Embedder
	}
	e := cfg.Embed
	switch e.Provider {
	case "openai":
		return embed.NewAPIEmbedder(embed.APIConfig{
			APIURL: e.APIURL, APIKey: e.APIKey, Model: e.Model, Dim: cfg.Dim, Timeout: e.Timeout,
			MaxRetries: e.MaxRetries, BatchSize: e.BatchSize, SendDimensions: e.SendDimensions,
		})
	case "offline":
		return embed.NewOfflineEmbedder(cfg.Dim)
	default:
		// llama.cpp is the default: point Embed.APIURL at your
		// llama-server (e.g. http://127.0.0.1:8080). See the README.
		return embed.NewLlamaCppEmbedder(embed.LlamaCppConfig{
			ServerURL: e.APIURL, Model: e.Model, Dim: cfg.Dim, APIKey: e.APIKey, Timeout: e.Timeout,
			MaxRetries: e.MaxRetries, BatchSize: e.BatchSize, Logger: lg,
		})
	}
}
