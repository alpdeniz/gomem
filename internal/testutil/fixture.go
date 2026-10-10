package testutil

import (
	"fmt"
	"math/rand"

	"github.com/alpdeniz/gomem"
)

// Fixture generates reproducible test items.
type Fixture struct {
	rng *rand.Rand
}

// NewFixture builds a fixture with a deterministic RNG.
func NewFixture(seed int64) *Fixture {
	return &Fixture{rng: rand.New(rand.NewSource(seed))}
}

var subjects = []string{
	"database", "query", "index", "cache", "authentication",
	"token", "password", "encryption", "latency", "throughput",
	"connection", "pool", "transaction", "rollback", "commit",
}

var verbs = []string{
	"optimize", "tune", "scale", "monitor", "debug",
	"profile", "benchmark", "compress", "partition", "shard",
}

// Item builds a single synthetic item with the given ordinal.
func (f *Fixture) Item(i int) gomem.Item {
	subj := subjects[f.rng.Intn(len(subjects))]
	verb := verbs[f.rng.Intn(len(verbs))]
	kind := gomem.KindFact
	switch f.rng.Intn(4) {
	case 0:
		kind = gomem.KindLesson
	case 1:
		kind = gomem.KindSkill
	case 3:
		kind = gomem.KindTemplate
	}
	title := fmt.Sprintf("%s %s #%d", verb, subj, i)
	body := fmt.Sprintf(
		"How to %s the %s subsystem. Steps: measure baseline, apply change, "+
			"verify %s improves. Caveats: watch for regressions in related %s paths.",
		verb, subj, subj, subj)
	tags := []string{subj}
	if f.rng.Intn(2) == 0 {
		tags = append(tags, verb)
	}
	conf := 0.5 + f.rng.Float64()*0.5
	return gomem.Item{
		Kind:  kind,
		Title: title,
		Body:  body,
		Tags:  tags,
		Provenance: gomem.Provenance{
			Source:     "run",
			RunID:      fmt.Sprintf("run-%d", i),
			Confidence: gomem.Confidence(conf),
		},
	}
}

// Items builds n synthetic items.
func (f *Fixture) Items(n int) []gomem.Item {
	out := make([]gomem.Item, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, f.Item(i))
	}
	return out
}
