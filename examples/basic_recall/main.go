// Command basic_recall is a minimal end-to-end demonstration of the
// gomem library: open a database, remember a handful of items, recall
// the most relevant ones for a goal, print stats, and close.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"os"
	"time"

	"github.com/alpdeniz/gomem"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Use a throwaway database in the temp dir.
	path := os.TempDir() + "/gomem-basic-recall-demo.db"
	for _, suffix := range []string{"", "-wal", "-shm"} {
		os.Remove(path + suffix)
	}

	cfg := gomem.DefaultConfig()
	cfg.Path = path
	cfg.Embed.Provider = "offline" // llama.cpp is the default; this keeps the demo zero-server
	cfg.Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	mem, err := gomem.New(ctx, cfg)
	if err != nil && !errors.Is(err, gomem.ErrCorrupt) {
		return fmt.Errorf("open: %w", err)
	}
	defer mem.Close()

	fmt.Println("== remembering items ==")
	items := []gomem.Item{
		{
			Kind:  gomem.KindFact,
			Title: "Composite indexes speed up filtered queries",
			Body:  "Adding a composite index aligned with WHERE and ORDER BY clauses avoids table scans and sorts.",
			Tags:  []string{"database", "index"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "demo-run",
				Confidence: gomem.Confidence(0.9),
			},
		},
		{
			Kind:  gomem.KindLesson,
			Title: "Cache hot reads to cut redundant work",
			Body:  "Memoizing frequently read, slowly changing results reduces repeated computation and lowers latency.",
			Tags:  []string{"cache", "performance"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "demo-run",
				Confidence: gomem.Confidence(0.8),
			},
		},
		{
			Kind:  gomem.KindSkill,
			Title: "Profile before optimizing",
			Body:  "Measure where time actually goes with a profiler; optimize the measured hotspot rather than guessing.",
			Tags:  []string{"profiling", "performance"},
			Provenance: gomem.Provenance{
				Source:     "human",
				Confidence: gomem.Confidence(0.7),
			},
		},
		{
			Kind:  gomem.KindFact,
			Title: "Connection pooling limits setup cost",
			Body:  "Reusing a bounded pool of database connections amortizes expensive connect/auth handshakes across many queries.",
			Tags:  []string{"database", "connection"},
			Provenance: gomem.Provenance{
				Source:     "run",
				RunID:      "demo-run",
				Confidence: gomem.Confidence(0.85),
			},
		},
	}
	for i := range items {
		id, err := mem.Remember(ctx, items[i])
		if err != nil {
			return fmt.Errorf("remember %q: %w", items[i].Title, err)
		}
		fmt.Printf("  stored %-45s id=%s\n", items[i].Title, id)
	}

	fmt.Println("\n== recalling for goal: 'speed up slow database queries' ==")
	hits, err := mem.Recall(ctx, gomem.RecallQuery{
		Goal: "speed up slow database queries",
		TopK: 5,
	})
	if err != nil {
		return fmt.Errorf("recall: %w", err)
	}
	if len(hits) == 0 {
		fmt.Println("  (no hits)")
	}
	for rank, h := range hits {
		fmt.Printf("  #%d score=%.3f [%s] %s\n", rank+1, h.Score, h.Kind, h.Title)
	}

	fmt.Println("\n== stats ==")
	st, err := mem.Stats(ctx)
	if err != nil {
		return fmt.Errorf("stats: %w", err)
	}
	fmt.Printf("  total items: %d\n", st.TotalItems)
	for k, c := range st.ByKind {
		fmt.Printf("    kind %-10s: %d\n", k, c)
	}
	fmt.Printf("  db size: %d bytes\n", st.DBSizeBytes)

	fmt.Println("\ndone.")
	return nil
}
