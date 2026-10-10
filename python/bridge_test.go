package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/alpdeniz/gomem"
)

func TestRunJSON_RecoversPanic(t *testing.T) {
	b := runJSON(func() (any, error) { panic("boom") })
	var doc map[string]string
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("invalid JSON %q: %v", b, err)
	}
	if doc["error"] != "panic: boom" {
		t.Fatalf("error = %q", doc["error"])
	}
}

func TestRunJSON_Result(t *testing.T) {
	if got := string(runJSON(func() (any, error) { return map[string]int{"n": 1}, nil })); got != `{"n":1}` {
		t.Fatalf("got %s", got)
	}
}

func overlayFrom(t *testing.T, raw string) gomem.Config {
	t.Helper()
	var doc configDoc
	if err := json.Unmarshal([]byte(raw), &doc); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	cfg := gomem.DefaultConfig()
	overlay(doc, &cfg)
	return cfg
}

func TestOverlay_ExplicitZeroApplied(t *testing.T) {
	cfg := overlayFrom(t, `{"alpha_dense":0,"max_retries":0,"gamma_salience":0,"w_frequency":0}`)
	if cfg.AlphaDense != 0 || cfg.GammaSalience != 0 || cfg.Embed.MaxRetries != 0 || cfg.WFrequency != 0 {
		t.Fatalf("explicit zeros not applied: %+v", cfg)
	}
	if cfg.BetaSparse != 0.3 {
		t.Fatalf("absent beta_sparse changed: %v", cfg.BetaSparse)
	}
}

func TestOverlay_MissingKeepsDefaults(t *testing.T) {
	def := gomem.DefaultConfig()
	cfg := overlayFrom(t, `{"path":"x.db","top_k":5}`)
	if cfg.Path != "x.db" || cfg.DefaultTopK != 5 {
		t.Fatalf("provided fields not applied: %+v", cfg)
	}
	if cfg.AlphaDense != def.AlphaDense || cfg.Embed.MaxRetries != def.Embed.MaxRetries ||
		cfg.Dim != def.Dim || cfg.Embed.Timeout != def.Embed.Timeout {
		t.Fatalf("defaults not preserved: %+v", cfg)
	}
}

func TestRememberMany_Bridge(t *testing.T) {
	cfg := gomem.DefaultConfig()
	cfg.Path = t.TempDir() + "/b.db"
	cfg.Embed.Provider = "offline"
	m, err := gomem.New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	out, err := rememberMany(context.Background(), m, `[{"id":"a","kind":"fact","title":"one","provenance":{"source":"run","confidence":0.8}},
	                              {"id":"b","kind":"fact","title":"two","provenance":{"source":"run","confidence":0.8}}]`)
	if err != nil {
		t.Fatal(err)
	}
	if ids := out.(map[string]any)["ids"].([]string); len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Fatalf("ids = %v", ids)
	}
	// An invalid item fails the whole batch.
	if _, err := rememberMany(context.Background(), m, `[{"id":"c","kind":"fact","title":"ok"},{"kind":"fact"}]`); err == nil {
		t.Fatal("expected error")
	}
	if _, err := m.Get(context.Background(), "c"); err == nil {
		t.Fatal("item c written despite batch failure")
	}
}

// TestRemember_TierDefaultAndExplicitZero guards decision D1 on the bridge:
// a wire item without "tier" is stored hot (2), an explicit "tier":2
// round-trips as 2, and an explicit "tier":0 stays cold.
func TestRemember_TierDefaultAndExplicitZero(t *testing.T) {
	cfg := gomem.DefaultConfig()
	cfg.Path = t.TempDir() + "/tier.db"
	cfg.Embed.Provider = "offline"
	m, err := gomem.New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	out, err := rememberMany(context.Background(), m, `[
	  {"id":"absent","kind":"fact","title":"no tier","provenance":{"source":"run","confidence":0.8}},
	  {"id":"py-default","kind":"fact","title":"python default","tier":2,"provenance":{"source":"run","confidence":0.8}},
	  {"id":"cold","kind":"fact","title":"explicit cold","tier":0,"provenance":{"source":"run","confidence":0.8}},
	  {"id":"warm","kind":"fact","title":"explicit warm","tier":1,"provenance":{"source":"run","confidence":0.8}}]`)
	if err != nil {
		t.Fatal(err)
	}
	if ids := out.(map[string]any)["ids"].([]string); len(ids) != 4 {
		t.Fatalf("ids = %v", ids)
	}
	for id, want := range map[string]int{"absent": 2, "py-default": 2, "cold": 0, "warm": 1} {
		it, err := m.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if it.Tier == nil || *it.Tier != want {
			t.Errorf("item %s: stored tier = %v, want %d", id, it.Tier, want)
		}
	}

	// The output document always carries the tier, including an explicit 0.
	it, _ := m.Get(context.Background(), "cold")
	b, err := json.Marshal(toWireItem(it))
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if v, ok := doc["tier"]; !ok || v != float64(0) {
		t.Fatalf("wire tier for cold item = %v (present=%v), want 0", v, ok)
	}
}

func TestOverlay_SendDimensionsAndEmbedLength(t *testing.T) {
	cfg := overlayFrom(t, `{"send_dimensions":false,"embed_length_threshold":0}`)
	if cfg.Embed.SendDimensions == nil || *cfg.Embed.SendDimensions {
		t.Fatalf("send_dimensions = %v", cfg.Embed.SendDimensions)
	}
	if cfg.EmbedLengthThreshold != 0 {
		t.Fatalf("embed_length_threshold = %d", cfg.EmbedLengthThreshold)
	}
	def := overlayFrom(t, `{}`)
	if def.Embed.SendDimensions != nil {
		t.Fatalf("default send_dimensions = %v", def.Embed.SendDimensions)
	}
	if def.EmbedLengthThreshold != 10 {
		t.Fatalf("default embed_length_threshold = %d", def.EmbedLengthThreshold)
	}
}

func TestOverlay_DenseIndexAndFusionMode(t *testing.T) {
	if got := overlayFrom(t, `{"fusion_mode":"rrf"}`).FusionMode; got != "rrf" {
		t.Fatalf("fusion_mode = %q", got)
	}
	if got := overlayFrom(t, `{}`).FusionMode; got != "legacy" {
		t.Fatalf("default fusion_mode = %q", got)
	}
	if got := overlayFrom(t, `{"dense_index":"memory"}`).DenseIndex; got != "memory" {
		t.Fatalf("dense_index = %q", got)
	}
	if got := overlayFrom(t, `{}`).DenseIndex; got != "sqlite" {
		t.Fatalf("default dense_index = %q", got)
	}
}

// TestRemember_ConfidenceAbsentAndExplicitZero guards decision D2 on the
// bridge: an absent "confidence" is stored as 1, an explicit 0 is kept, and
// the output document carries the 0.
func TestRemember_ConfidenceAbsentAndExplicitZero(t *testing.T) {
	cfg := gomem.DefaultConfig()
	cfg.Path = t.TempDir() + "/conf.db"
	cfg.Embed.Provider = "offline"
	m, err := gomem.New(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if _, err := rememberMany(context.Background(), m, `[
	  {"id":"absent","kind":"fact","title":"no confidence","provenance":{"source":"run"}},
	  {"id":"zero","kind":"fact","title":"untrusted","provenance":{"source":"run","confidence":0}},
	  {"id":"half","kind":"fact","title":"half","provenance":{"source":"run","confidence":0.5}}]`); err != nil {
		t.Fatal(err)
	}
	for id, want := range map[string]float64{"absent": 1, "zero": 0, "half": 0.5} {
		it, err := m.Get(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		if it.Provenance.Confidence == nil || *it.Provenance.Confidence != want {
			t.Errorf("item %s: confidence = %v, want %v", id, it.Provenance.Confidence, want)
		}
	}
	it, _ := m.Get(context.Background(), "zero")
	b, err := json.Marshal(toWireItem(it))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Provenance map[string]any `json:"provenance"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if v, ok := doc.Provenance["confidence"]; !ok || v != float64(0) {
		t.Fatalf("wire confidence for untrusted item = %v (present=%v), want 0", v, ok)
	}
}

func handleCount() int {
	handlesMu.Lock()
	defer handlesMu.Unlock()
	return len(handles)
}

func TestOpenMemory_CorruptReturnsHandleAndError(t *testing.T) {
	ctx := context.Background()
	path := t.TempDir() + "/c.db"
	cfg := gomem.DefaultConfig()
	cfg.Path = path
	cfg.Embed.Provider = "offline"

	m, err := gomem.New(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "seed"}); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) < 16 {
		t.Fatalf("db too small to corrupt: %d", len(data))
	}
	for i := 0; i < 16; i++ {
		data[i] ^= 0xFF
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	before := handleCount()
	out, err := openMemory(ctx, cfg)
	if err != nil {
		t.Fatalf("openMemory: %v", err)
	}
	doc, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("out type %T", out)
	}
	h, ok := doc["handle"].(uintptr)
	if !ok || h == 0 {
		t.Fatalf("handle = %v", doc["handle"])
	}
	msg, _ := doc["error"].(string)
	if !strings.Contains(strings.ToLower(msg), "corrupt") {
		t.Fatalf("error = %q", msg)
	}
	got, ok := acquire(h)
	if !ok || got == nil {
		t.Fatal("handle not registered")
	}
	defer func() {
		_ = got.Close()
		deregister(h)
	}()
	if handleCount() != before+1 {
		t.Fatalf("handles = %d, want %d", handleCount(), before+1)
	}
	st, err := got.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if st.TotalItems != 0 {
		t.Fatalf("recreated store has %d items", st.TotalItems)
	}
}

func TestOpenMemory_InvalidConfigNoHandle(t *testing.T) {
	before := handleCount()
	out, err := openMemory(context.Background(), gomem.Config{Path: t.TempDir() + "/x.db"})
	if err == nil || !errors.Is(err, gomem.ErrInvalid) {
		t.Fatalf("err = %v", err)
	}
	if out != nil {
		t.Fatalf("out = %v", out)
	}
	if handleCount() != before {
		t.Fatalf("leaked handle: %d -> %d", before, handleCount())
	}
}
