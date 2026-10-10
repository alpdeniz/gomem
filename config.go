package gomem

import (
	"cmp"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/alpdeniz/gomem/internal/embed"
)

// Embedder turns text into fixed-size vectors. Implement it and set
// Config.Embedder to plug in a custom embedding backend.
type Embedder = embed.Embedder

// Config holds every tunable for a Memory instance. Start from
// DefaultConfig and override what you need; a bare Config{} is not
// valid.
//
// Path must name a file: an in-memory database (":memory:") fails with
// ErrInvalid.
//
// New replaces a zero (or non-positive) value with its DefaultConfig value
// for: Path, Dim, DefaultTopK, RecencyHalfLifeDays, DedupJaccardThreshold,
// MaxItems, MaxPayloadKB, FreqCap, EvictionAgeDays, Embed.Provider,
// DenseIndex and FusionMode. The weights are NOT defaulted: AlphaDense,
// BetaSparse and GammaSalience must sum to more than zero, and so must
// WConfidence, WFrequency and WRecency; otherwise New fails with an error
// wrapping ErrInvalid. An unknown Embed.Provider, DenseIndex or FusionMode
// also fails with ErrInvalid. Embed.APIKey and Embed.APIURL fall back to
// the GOMEM_EMBED_API_KEY and GOMEM_EMBED_API_URL environment variables
// when empty.
type Config struct {
	// Storage.
	Path string // file path to the .db (parent dirs auto-created)
	Dim  int    // embedding dimensionality (default 512)

	// Embedding.
	Embed EmbedCfg

	// EmbedLengthThreshold is the minimum length (in runes) of an item's
	// embed text (title + body) for it to be embedded. Shorter items are
	// stored FTS-only (sparse recall still finds them). DefaultConfig
	// sets this to 10; 0 embeds everything.
	EmbedLengthThreshold int

	// Embedder optionally overrides the embedder selected from Embed
	// (when non-nil it is used verbatim). Primarily useful for tests and
	// bespoke integrations; the default and the OpenAI-compatible adapter
	// cover ordinary use.
	Embedder Embedder

	// Retrieval defaults.
	DefaultTopK           int
	RecencyHalfLifeDays   float64
	DedupJaccardThreshold float64

	// DenseIndex selects how the dense leg reads vectors: "sqlite" (the
	// default) scans them from the database on every query; "memory" keeps
	// an in-RAM copy (N*dim*4 bytes) for much faster recall at large sizes.
	DenseIndex string

	// FusionMode selects how the dense and sparse leg scores are put on a
	// common [0,1] scale before weighting: "legacy" (the default: dense
	// (cos+1)/2, sparse min-max), "minmax" (both legs min-max), or "rrf"
	// (reciprocal rank, 1/(60+rank) per leg).
	FusionMode string

	// Fusion weights (relative importance of each recall leg). They must
	// sum to more than zero.
	AlphaDense    float64
	BetaSparse    float64
	GammaSalience float64

	// Salience weights: the computed (unstored) salience orders capacity
	// eviction and feeds the recall score. They must sum to more than zero.
	WConfidence float64
	WFrequency  float64
	WRecency    float64
	FreqCap     int

	// Housekeeping. EvictionAgeDays: Reflect evicts an item idle (time
	// since last access) for longer than this, regardless of its
	// confidence or access count. Default 90.
	EvictionAgeDays int

	// Limits. MaxItems is a soft cap enforced by Reflect (lowest
	// salience evicted first), not by Remember.
	MaxItems     int
	MaxPayloadKB int

	// Logging.
	Logger *slog.Logger
}

// EmbedCfg configures the pluggable embedder.
type EmbedCfg struct {
	// Provider selects the embedder: "llama.cpp" (DEFAULT — a local
	// llama-server, true semantics, no cloud), "offline" (zero-network
	// feature-hash), or "openai" (OpenAI-compatible HTTP). Matching is
	// case-insensitive, and "llamacpp"/"llama", "hash" and "api" are
	// aliases; any other value makes New fail with ErrInvalid.
	Provider string
	// APIURL is the server endpoint: for "llama.cpp" the llama-server base
	// URL (e.g. http://127.0.0.1:8080); for "openai" the /embeddings URL.
	APIURL string
	// APIKey is the embedding model's credential, sent to the
	// server as "Authorization: Bearer <key>". Applies to the
	// llama.cpp and openai providers; leave empty for an
	// unauthenticated local server.
	APIKey     string
	Model      string
	Timeout    time.Duration
	MaxRetries int
	BatchSize  int
	CacheDir   string // "" = in-memory only; else a disk cache dir
	// SendDimensions controls whether the openai provider sends the
	// "dimensions" request field. nil (the default) sends it; set it to
	// a pointer to false for compatible servers that reject unknown fields.
	SendDimensions *bool
}

// DefaultConfig returns a Config populated with the recommended
// defaults (512-dim, top-20, 30-day half-life, balanced fusion/salience
// weights).
func DefaultConfig() Config {
	return Config{
		Path:                  "gomem.db",
		Dim:                   512,
		EmbedLengthThreshold:  10,
		DefaultTopK:           20,
		RecencyHalfLifeDays:   30,
		DedupJaccardThreshold: 0.55,
		AlphaDense:            0.5,
		BetaSparse:            0.3,
		GammaSalience:         0.2,
		WConfidence:           0.2,
		WFrequency:            0.3,
		WRecency:              0.5,
		FreqCap:               100,
		EvictionAgeDays:       90,
		MaxItems:              200000,
		MaxPayloadKB:          64,
		DenseIndex:            "sqlite",
		FusionMode:            "legacy",
		Embed: EmbedCfg{
			// llama.cpp is the DEFAULT embedder so recall is genuinely
			// semantic out of the box (a local llama-server, no cloud hop).
			// APIURL and APIKey are left blank: validate() reads them from
			// GOMEM_EMBED_API_URL and GOMEM_EMBED_API_KEY (the URL falls
			// back to the llama-server default, 127.0.0.1:8080). For the
			// zero-network feature-hash embedder, set Provider to "offline".
			Provider:   "llama.cpp",
			Timeout:    30 * time.Second,
			MaxRetries: 3,
			BatchSize:  32,
		},
	}
}

// EnvAPIKey is the environment variable consulted for the embedding
// credential when none is set explicitly on the config.
const EnvAPIKey = "GOMEM_EMBED_API_KEY"

// EnvAPIURL is the environment variable consulted for the embedding
// server URL when none is set explicitly on the config.
const EnvAPIURL = "GOMEM_EMBED_API_URL"

// providerAliases maps alternative Embed.Provider spellings to their
// canonical name.
var providerAliases = map[string]string{
	"llamacpp": "llama.cpp", "llama": "llama.cpp", "hash": "offline", "api": "openai",
}

// positive returns v, or def when v is not positive.
func positive[T ~int | ~float64](v, def T) T {
	if v > 0 {
		return v
	}
	return def
}

// oneOf returns v, or the first allowed value when v is empty, and fails
// with ErrInvalid when v is not allowed.
func oneOf(name, v string, allowed ...string) (string, error) {
	if v == "" {
		return allowed[0], nil
	}
	if !slices.Contains(allowed, v) {
		return "", fmt.Errorf("%w: unknown %s %q (want one of %q)", ErrInvalid, name, v, allowed)
	}
	return v, nil
}

// validate normalizes zero-value fields to defaults and enforces
// invariants. It mutates the receiver in place.
func (c *Config) validate() error {
	// The embedding credential and server URL come from the environment
	// when not set explicitly (programmatically or via the bridge), keeping
	// secrets and host-specific addresses out of the source tree.
	c.Embed.APIKey = cmp.Or(c.Embed.APIKey, os.Getenv(EnvAPIKey))
	c.Embed.APIURL = cmp.Or(c.Embed.APIURL, os.Getenv(EnvAPIURL))
	d := DefaultConfig()
	c.Path = cmp.Or(c.Path, d.Path)
	// Every pooled connection would open its own empty in-memory database.
	if strings.HasPrefix(c.Path, ":memory:") || strings.Contains(c.Path, "mode=memory") {
		return fmt.Errorf("%w: in-memory databases are not supported; use a temp file", ErrInvalid)
	}
	c.Dim = positive(c.Dim, d.Dim)
	c.DefaultTopK = positive(c.DefaultTopK, d.DefaultTopK)
	c.RecencyHalfLifeDays = positive(c.RecencyHalfLifeDays, d.RecencyHalfLifeDays)
	c.DedupJaccardThreshold = positive(c.DedupJaccardThreshold, d.DedupJaccardThreshold)
	c.MaxItems = positive(c.MaxItems, d.MaxItems)
	c.MaxPayloadKB = positive(c.MaxPayloadKB, d.MaxPayloadKB)
	c.FreqCap = positive(c.FreqCap, d.FreqCap)
	c.EvictionAgeDays = positive(c.EvictionAgeDays, d.EvictionAgeDays)
	var err error
	provider := strings.ToLower(strings.TrimSpace(c.Embed.Provider))
	provider = cmp.Or(providerAliases[provider], provider)
	if c.Embed.Provider, err = oneOf("Embed.Provider", provider, "llama.cpp", "offline", "openai"); err != nil {
		return err
	}
	if c.DenseIndex, err = oneOf("DenseIndex", c.DenseIndex, "sqlite", "memory"); err != nil {
		return err
	}
	if c.FusionMode, err = oneOf("FusionMode", c.FusionMode, "legacy", "minmax", "rrf"); err != nil {
		return err
	}
	if c.AlphaDense+c.BetaSparse+c.GammaSalience <= 0 {
		return fmt.Errorf("%w: fusion weights must sum greater than zero", ErrInvalid)
	}
	if c.WConfidence+c.WFrequency+c.WRecency <= 0 {
		return fmt.Errorf("%w: salience weights must sum greater than zero", ErrInvalid)
	}
	return nil
}
