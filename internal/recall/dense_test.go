package recall

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/embed"
)

// refCosine is the straightforward cosine similarity used as the oracle.
func refCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		af, bf := float64(a[i]), float64(b[i])
		dot += af * bf
		na += af * af
		nb += bf * bf
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestDenseKNN_TopKMatchesFullSort compares the bounded top-k buffer to a
// full sort over random vectors.
func TestDenseKNN_TopKMatchesFullSort(t *testing.T) {
	const n, dim, k = 1000, 32, 30
	ctx := context.Background()
	env := newTestEnv(t, embed.NewOfflineEmbedder(dim))
	rng := rand.New(rand.NewSource(1))
	randVec := func() []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}

	now := time.Now().UTC().Format(time.RFC3339)
	tx, err := env.d.Raw.BeginTx(ctx, nil)
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	vecs := map[string][]float32{}
	for i := 0; i < n; i++ {
		id := "v-" + itoa(i)
		vecs[id] = randVec()
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
			 VALUES(?, 'fact', 't', '', '{}', 2, ?, ?)`, id, now, now); err != nil {
			t.Fatalf("insert item: %v", err)
		}
		if _, err := tx.ExecContext(ctx, "INSERT INTO vec(item_id, embedding) VALUES(?,?)",
			id, embed.EncodeVec(vecs[id])); err != nil {
			t.Fatalf("insert vec: %v", err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}

	q := randVec()
	type scored struct {
		id  string
		sim float64
	}
	want := make([]scored, 0, n)
	for id, v := range vecs {
		want = append(want, scored{id, refCosine(q, v)})
	}
	sort.Slice(want, func(i, j int) bool {
		if want[i].sim != want[j].sim {
			return want[i].sim > want[j].sim
		}
		return want[i].id < want[j].id
	})

	where, args := filterSQL(Query{}, "i")
	got, err := denseKNN(ctx, env.d, q, k, where, args)
	if err != nil {
		t.Fatalf("denseKNN: %v", err)
	}
	if len(got) != k {
		t.Fatalf("got %d hits, want %d", len(got), k)
	}
	for i, h := range got {
		if h.ID != want[i].id || h.Score != want[i].sim {
			t.Fatalf("rank %d: got %s/%v, want %s/%v", i, h.ID, h.Score, want[i].id, want[i].sim)
		}
	}
}
