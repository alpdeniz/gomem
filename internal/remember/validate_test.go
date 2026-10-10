package remember

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/errs"
	"github.com/alpdeniz/gomem/internal/model"
)

func TestNormalize_Valid(t *testing.T) {
	now := time.Date(2026, 10, 4, 12, 0, 0, 0, time.UTC)
	in := Input{
		Kind:  "fact",
		Title: "  A titled item  ",
		Body:  "body",
		Tags:  []string{" Auth ", "AUTH", "", "db"},
		Provenance: model.ProvDoc{
			Source:     "run",
			Confidence: 0.9,
		},
	}
	out, err := Normalize(in, NormalizeOpts{Now: now, MaxPayloadKB: 64})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.ID == "" {
		t.Fatal("expected a generated ULID")
	}
	if len(out.ID) != 26 {
		t.Fatalf("expected 26-char ID, got %q", out.ID)
	}
	if out.Title != "A titled item" {
		t.Fatalf("expected trimmed title, got %q", out.Title)
	}
	// Tags: lowercased, trimmed, deduped, empties dropped.
	want := []string{"auth", "db"}
	if len(out.Tags) != len(want) {
		t.Fatalf("expected tags %v, got %v", want, out.Tags)
	}
	for i := range want {
		if out.Tags[i] != want[i] {
			t.Fatalf("tag[%d]=%q, want %q", i, out.Tags[i], want[i])
		}
	}
	// Provenance defaults.
	if out.Provenance.CreatedAt.IsZero() {
		t.Fatal("expected CreatedAt defaulted")
	}
	if out.Provenance.Confidence != 0.9 {
		t.Fatalf("expected confidence preserved, got %v", out.Provenance.Confidence)
	}
}

func TestNormalize_ProvenanceDefaults(t *testing.T) {
	now := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	in := Input{Kind: "fact", Title: "t"}
	out, err := Normalize(in, NormalizeOpts{Now: now})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.Provenance.Source != "run" {
		t.Fatalf("expected source defaulted to run, got %q", out.Provenance.Source)
	}
}

// Confidence arrives resolved from the public layer: 0 is a real value, and
// out-of-range values are clamped.
func TestNormalize_ConfidenceClampAndNaN(t *testing.T) {
	now := time.Now().UTC()
	for in, want := range map[float64]float64{-3: 0, 0: 0, 0.4: 0.4, 1: 1, 9: 1} {
		out, err := Normalize(Input{Kind: "fact", Title: "t", Provenance: model.ProvDoc{Confidence: in}}, NormalizeOpts{Now: now})
		if err != nil || out.Provenance.Confidence != want {
			t.Fatalf("confidence %v: got %v, %v; want %v", in, out.Provenance.Confidence, err, want)
		}
	}
	if _, err := Normalize(Input{Kind: "fact", Title: "t", Provenance: model.ProvDoc{Confidence: math.NaN()}}, NormalizeOpts{Now: now}); !errors.Is(err, errs.ErrInvalid) {
		t.Fatalf("NaN confidence: err = %v, want errs.ErrInvalid", err)
	}
}

func TestNormalize_TierClamp(t *testing.T) {
	now := time.Now().UTC()
	for in, want := range map[int]int{-5: 0, 0: 0, 1: 1, 2: 2, 7: 2} {
		in := in
		out, err := Normalize(Input{Kind: "fact", Title: "t", Tier: &in}, NormalizeOpts{Now: now})
		if err != nil {
			t.Fatalf("in=%d: unexpected error: %v", in, err)
		}
		if out.Tier == nil || *out.Tier != want {
			t.Fatalf("tier in=%d: got %v, want %d", in, out.Tier, want)
		}
	}
	out, err := Normalize(Input{Kind: "fact", Title: "t"}, NormalizeOpts{Now: now})
	if err != nil || out.Tier != nil {
		t.Fatalf("unspecified tier must stay nil: %v %v", out.Tier, err)
	}
}

func TestNormalize_Errors(t *testing.T) {
	now := time.Now().UTC()
	cases := []struct {
		name string
		in   Input
	}{
		{"empty kind", Input{Title: "t"}},
		{"empty title", Input{Kind: "fact"}},
		{"empty id and empty kind", Input{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Normalize(tc.in, NormalizeOpts{Now: now})
			if !errors.Is(err, errs.ErrInvalid) {
				t.Fatalf("expected ErrInvalid, got %v", err)
			}
		})
	}
}

func TestNormalize_OversizedPayload(t *testing.T) {
	now := time.Now().UTC()
	big := make([]byte, 65*1024) // 65KB > 64KB cap
	for i := range big {
		big[i] = 'x'
	}
	_, err := Normalize(Input{Kind: "fact", Title: "t", Payload: big}, NormalizeOpts{Now: now, MaxPayloadKB: 64})
	if !errors.Is(err, errs.ErrInvalid) {
		t.Fatalf("expected ErrInvalid for oversized payload, got %v", err)
	}
}
