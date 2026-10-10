package remember

import (
	"math"
	"strings"
	"time"

	"github.com/alpdeniz/gomem/internal/errs"
	"github.com/alpdeniz/gomem/internal/textutil"
)

// NormalizeOpts carries the context needed to normalize an Input.
type NormalizeOpts struct {
	MaxPayloadKB int
	Now          time.Time
}

// Normalize validates and fills defaults on an Input, returning the
// cleaned value. It reports errs.ErrInvalid for structural problems
// (missing kind/title, oversized payload).
func Normalize(in Input, opts NormalizeOpts) (Input, error) {
	if opts.Now.IsZero() {
		opts.Now = time.Now().UTC()
	}
	if in.ID == "" {
		in.ID = NewULID()
	}
	in.ID = strings.TrimSpace(in.ID)
	in.Title = strings.TrimSpace(in.Title)
	if in.ID == "" || strings.TrimSpace(in.Kind) == "" || in.Title == "" {
		return in, errs.ErrInvalid
	}
	if opts.MaxPayloadKB > 0 && len(in.Payload) > opts.MaxPayloadKB*1024 {
		return in, errs.ErrInvalid
	}
	p := &in.Provenance
	if p.Source == "" {
		p.Source = "run"
	}
	// Confidence arrives resolved (the public layer maps "unspecified" to
	// 1), so 0 is a real, untrusted value here.
	if math.IsNaN(p.Confidence) {
		return in, errs.ErrInvalid
	}
	p.Confidence = min(max(p.Confidence, 0), 1)
	if p.CreatedAt.IsZero() {
		p.CreatedAt = opts.Now
	}
	if in.Tier != nil {
		t := min(max(*in.Tier, 0), 2)
		in.Tier = &t // copy: never mutate the caller's value
	}
	in.Tags = textutil.CleanTags(in.Tags)
	in.Supersedes = strings.TrimSpace(in.Supersedes)
	if in.Supersedes == in.ID {
		return in, errs.ErrInvalid
	}
	if in.EmbedText == "" {
		in.EmbedText = in.Title + "\n" + in.Body
	}
	return in, nil
}
