package gomem

import (
	"errors"
	"strings"
	"testing"
)

func TestValidate_FusionWeightsMustBePositive(t *testing.T) {
	c := DefaultConfig()
	c.AlphaDense, c.BetaSparse, c.GammaSalience = 0, 0, 0
	err := c.validate()
	if err == nil {
		t.Fatal("expected an error for all-zero fusion weights")
	}
	if !strings.Contains(err.Error(), "fusion weights") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidate_SalienceWeightsMustBePositive(t *testing.T) {
	c := DefaultConfig()
	c.WConfidence, c.WFrequency, c.WRecency = 0, 0, 0
	err := c.validate()
	if err == nil {
		t.Fatal("expected an error for all-zero salience weights")
	}
	if !strings.Contains(err.Error(), "salience weights") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestValidate_ErrorsWrapErrInvalid confirms every validation failure,
// including a bare Config{}, is classifiable with errors.Is.
func TestValidate_ErrorsWrapErrInvalid(t *testing.T) {
	bad := map[string]Config{"zero config": {}}
	c := DefaultConfig()
	c.AlphaDense, c.BetaSparse, c.GammaSalience = 0, 0, 0
	bad["fusion weights"] = c
	c = DefaultConfig()
	c.WConfidence, c.WFrequency, c.WRecency = 0, 0, 0
	bad["salience weights"] = c
	c = DefaultConfig()
	c.DenseIndex = "hnsw"
	bad["dense index"] = c
	c = DefaultConfig()
	c.FusionMode = "borda"
	bad["fusion mode"] = c
	c = DefaultConfig()
	c.Embed.Provider = "ollama"
	bad["embed provider"] = c
	c = DefaultConfig()
	c.Path = ":memory:"
	bad["in-memory path"] = c
	c = DefaultConfig()
	c.Path = "file:x?mode=memory&cache=shared"
	bad["in-memory uri"] = c
	for name, cfg := range bad {
		if err := cfg.validate(); !errors.Is(err, ErrInvalid) {
			t.Fatalf("%s: expected an error wrapping ErrInvalid, got %v", name, err)
		}
	}
}

func TestValidate_ProviderAliasesAreCanonicalized(t *testing.T) {
	for in, want := range map[string]string{
		"": "llama.cpp", "LLaMA.cpp": "llama.cpp", "llamacpp": "llama.cpp",
		" offline ": "offline", "hash": "offline", "OpenAI": "openai", "api": "openai",
	} {
		c := DefaultConfig()
		c.Embed.Provider = in
		if err := c.validate(); err != nil {
			t.Fatalf("%q: %v", in, err)
		}
		if c.Embed.Provider != want {
			t.Fatalf("%q: got %q, want %q", in, c.Embed.Provider, want)
		}
	}
}

// TestValidate_ZeroValuesTakeDefaultConfig pins that New's fallbacks for
// zero fields are exactly the DefaultConfig values.
func TestValidate_ZeroValuesTakeDefaultConfig(t *testing.T) {
	c := Config{AlphaDense: 1, WRecency: 1}
	if err := c.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	d := DefaultConfig()
	if c.Path != d.Path || c.Dim != d.Dim || c.DefaultTopK != d.DefaultTopK ||
		c.RecencyHalfLifeDays != d.RecencyHalfLifeDays || c.DedupJaccardThreshold != d.DedupJaccardThreshold ||
		c.MaxItems != d.MaxItems || c.MaxPayloadKB != d.MaxPayloadKB || c.FreqCap != d.FreqCap ||
		c.EvictionAgeDays != d.EvictionAgeDays || c.Embed.Provider != d.Embed.Provider ||
		c.DenseIndex != d.DenseIndex || c.FusionMode != d.FusionMode {
		t.Fatalf("zero-value fallbacks differ from DefaultConfig:\n got %+v\nwant %+v", c, d)
	}
}

func TestValidate_DefaultConfigIsValid(t *testing.T) {
	c := DefaultConfig()
	if err := c.validate(); err != nil {
		t.Fatalf("expected the default config to validate, got: %v", err)
	}
}

// TestValidate_APIKeyFromEnv confirms the embedding credential is
// read from the environment when not set explicitly.
func TestValidate_APIKeyFromEnv(t *testing.T) {
	t.Setenv(EnvAPIKey, "env-secret")
	c := DefaultConfig()
	c.Embed.APIKey = "" // ensure we are exercising the env path
	if err := c.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if c.Embed.APIKey != "env-secret" {
		t.Fatalf("expected API key from env, got %q", c.Embed.APIKey)
	}
}

// TestValidate_ExplicitAPIKeyWinsOverEnv confirms an explicitly
// configured key is not clobbered by the environment value.
func TestValidate_ExplicitAPIKeyWinsOverEnv(t *testing.T) {
	t.Setenv(EnvAPIKey, "env-secret")
	c := DefaultConfig()
	c.Embed.APIKey = "explicit-key"
	if err := c.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if c.Embed.APIKey != "explicit-key" {
		t.Fatalf("expected the explicit key to win, got %q", c.Embed.APIKey)
	}
}

// TestDefaultConfig_NoHardcodedSecret confirms the secret no longer
// ships in the source: with the environment isolated, the default
// config carries an empty key.
func TestDefaultConfig_NoHardcodedSecret(t *testing.T) {
	t.Setenv(EnvAPIKey, "")
	c := DefaultConfig()
	if c.Embed.APIKey != "" {
		t.Fatalf("expected no hardcoded API key in the default config, got %q", c.Embed.APIKey)
	}
}

// TestValidate_APIURLFromEnv confirms the embedding endpoint is
// read from the environment when not set explicitly.
func TestValidate_APIURLFromEnv(t *testing.T) {
	t.Setenv(EnvAPIURL, "http://env-host:9000")
	c := DefaultConfig()
	c.Embed.APIURL = ""
	if err := c.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if c.Embed.APIURL != "http://env-host:9000" {
		t.Fatalf("expected API URL from env, got %q", c.Embed.APIURL)
	}
}

// TestValidate_ExplicitAPIURLWinsOverEnv confirms an explicitly
// configured URL is not clobbered by the environment value.
func TestValidate_ExplicitAPIURLWinsOverEnv(t *testing.T) {
	t.Setenv(EnvAPIURL, "http://env-host:9000")
	c := DefaultConfig()
	c.Embed.APIURL = "http://explicit:1234"
	if err := c.validate(); err != nil {
		t.Fatalf("validate: %v", err)
	}
	if c.Embed.APIURL != "http://explicit:1234" {
		t.Fatalf("expected the explicit URL to win, got %q", c.Embed.APIURL)
	}
}

func TestValidate_DenseIndex(t *testing.T) {
	for in, want := range map[string]string{"": "sqlite", "sqlite": "sqlite", "memory": "memory"} {
		c := DefaultConfig()
		c.DenseIndex = in
		if err := c.validate(); err != nil || c.DenseIndex != want {
			t.Fatalf("%q: got %q, err %v; want %q", in, c.DenseIndex, err, want)
		}
	}
	c := DefaultConfig()
	c.DenseIndex = "hnsw"
	if err := c.validate(); err == nil {
		t.Fatal("expected an error for an unknown DenseIndex")
	}
}

func TestValidate_FusionMode(t *testing.T) {
	for in, want := range map[string]string{"": "legacy", "legacy": "legacy", "minmax": "minmax", "rrf": "rrf"} {
		c := DefaultConfig()
		c.FusionMode = in
		if err := c.validate(); err != nil || c.FusionMode != want {
			t.Fatalf("%q: got %q, err %v; want %q", in, c.FusionMode, err, want)
		}
	}
	c := DefaultConfig()
	c.FusionMode = "borda"
	if err := c.validate(); err == nil {
		t.Fatal("expected an error for an unknown FusionMode")
	}
}
