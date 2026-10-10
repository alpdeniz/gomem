package gomem_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/alpdeniz/gomem"
)

func Example() {
	dir, err := os.MkdirTemp("", "gomem-example-")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	ctx := context.Background()
	cfg := gomem.DefaultConfig()
	cfg.Path = filepath.Join(dir, "agent.db")
	cfg.Embed.Provider = "offline"

	mem, err := gomem.New(ctx, cfg)
	if err != nil && !errors.Is(err, gomem.ErrCorrupt) {
		log.Fatal(err)
	}
	defer mem.Close()

	_, err = mem.Remember(ctx, gomem.Item{
		Kind:  gomem.KindFact,
		Title: "Composite indexes speed up filtered queries",
		Body:  "Aligning a composite index with WHERE and ORDER BY clauses avoids table scans and sorts.",
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	hits, err := mem.Recall(ctx, gomem.RecallQuery{
		Goal: "speed up slow database queries",
		TopK: 1,
	})
	if err != nil {
		log.Fatal(err)
	}
	if len(hits) > 0 {
		fmt.Println(hits[0].Title)
	}
	// Output:
	// Composite indexes speed up filtered queries
}
