package recall

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/alpdeniz/gomem/internal/embed"
)

const idxDim = 32

// newIndexEnv returns an env plus an engine over the same database with the
// in-memory index on.
func newIndexEnv(t *testing.T) (*testEnv, *Engine) {
	t.Helper()
	env := newTestEnv(t, embed.NewOfflineEmbedder(idxDim))
	mem := NewEngine(env.d, env.emb, nil, nil, EngineConfig{Dim: idxDim, DefaultTopK: 10, DenseIndex: "memory"})
	t.Cleanup(mem.Close)
	return env, mem
}

func randVec(rng *rand.Rand) []float32 {
	v := make([]float32, idxDim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// seedVec inserts an item with a vector (raw SQL, like the other tests).
func (e *testEnv) seedVec(t *testing.T, id, kind string, vec []float32) {
	t.Helper()
	now := time.Now().UTC().Format(time.RFC3339)
	if _, err := e.d.Raw.Exec(`INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
		VALUES(?, ?, 't', '', '{}', 2, ?, ?)`, id, kind, now, now); err != nil {
		t.Fatalf("insert item: %v", err)
	}
	if _, err := e.d.Raw.Exec("INSERT INTO vec(item_id, embedding) VALUES(?,?)", id, embed.EncodeVec(vec)); err != nil {
		t.Fatalf("insert vec: %v", err)
	}
}

func searchBoth(t *testing.T, env *testEnv, mem *Engine, q Query, qvec []float32, k int) (want, got []hit) {
	t.Helper()
	ctx := context.Background()
	where, args := filterSQL(q, "i")
	want, err := denseKNN(ctx, env.d, qvec, k, where, args)
	if err != nil {
		t.Fatalf("denseKNN: %v", err)
	}
	got, err = mem.idx.search(ctx, qvec, k, where, args)
	if err != nil {
		t.Fatalf("index search: %v", err)
	}
	return want, got
}

func TestDenseIndex_EquivalentToSQLite(t *testing.T) {
	env, mem := newIndexEnv(t)
	rng := rand.New(rand.NewSource(7))
	kinds := []string{"fact", "lesson", "skill"}
	for i := 0; i < 1500; i++ {
		kind := kinds[rng.Intn(len(kinds))]
		if i%500 == 0 {
			kind = "rare" // 3 items, far down every ranking
		}
		env.seedVec(t, "v-"+itoa(i), kind, randVec(rng))
	}
	env.insertItem(t, "tagged", "fact", "t", "b", []string{"x"}, 0.9)
	if _, err := env.d.Raw.Exec("UPDATE items SET deleted_at='2026-01-01T00:00:00Z' WHERE id IN ('v-1','v-2','v-3')"); err != nil {
		t.Fatal(err)
	}

	queries := []Query{{}, {Kinds: []string{"lesson"}}, {Kinds: []string{"rare"}}, {MinTier: 3}, {Tags: []string{"x"}}}
	for i := 0; i < 20; i++ {
		qvec := randVec(rng)
		for _, q := range queries {
			want, got := searchBoth(t, env, mem, q, qvec, 10)
			if len(want) != len(got) {
				t.Fatalf("query %d %+v: %d hits, want %d", i, q, len(got), len(want))
			}
			for j := range want {
				if want[j].ID != got[j].ID || math.Abs(want[j].Score-got[j].Score) > 1e-5 {
					t.Fatalf("query %d %+v rank %d: got %v, want %v", i, q, j, got[j], want[j])
				}
			}
		}
	}
}

func TestDenseIndex_SeesRawSQLWrites(t *testing.T) {
	env, mem := newIndexEnv(t)
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 20; i++ {
		env.seedVec(t, "v-"+itoa(i), "fact", randVec(rng))
	}
	qvec := randVec(rng)
	if _, got := searchBoth(t, env, mem, Query{}, qvec, 50); len(got) != 20 {
		t.Fatalf("got %d hits, want 20", len(got))
	}
	// Another connection commits; the next search must see it.
	env.seedVec(t, "late", "fact", qvec)
	_, got := searchBoth(t, env, mem, Query{}, qvec, 50)
	if len(got) != 21 || got[0].ID != "late" {
		t.Fatalf("late write not visible: %v", got)
	}
}

func TestDenseIndex_DeleteAndReflectInvalidate(t *testing.T) {
	ctx := context.Background()
	env, mem := newIndexEnv(t)
	rng := rand.New(rand.NewSource(2))
	qvec := randVec(rng)
	env.seedVec(t, "a", "fact", qvec)
	env.seedVec(t, "b", "fact", randVec(rng))
	searchBoth(t, env, mem, Query{}, qvec, 10)
	x := mem.idx

	// An own write goes through IndexWrite: the index is updated in place
	// and stays loaded and fresh (the commit is not mistaken for a foreign one).
	if _, err := mem.IndexWrite(ctx, func() ([]string, error) {
		_, err := env.d.Raw.Exec("UPDATE items SET deleted_at='2026-01-01T00:00:00Z' WHERE id='a'")
		return []string{"a"}, err
	}); err != nil {
		t.Fatal(err)
	}
	if _, ok := x.pos["a"]; ok || !x.loaded || !x.fresh(ctx) || len(x.ids) != 1 {
		t.Fatalf("delete not applied in place: loaded=%v ids=%v", x.loaded, x.ids)
	}
	// Reviving an item through IndexWrite re-adds it from the database.
	if _, err := mem.IndexWrite(ctx, func() ([]string, error) {
		_, err := env.d.Raw.Exec("UPDATE items SET deleted_at=NULL WHERE id='a'")
		return []string{"a"}, err
	}); err != nil {
		t.Fatal(err)
	}
	if _, ok := x.pos["a"]; !ok || !x.fresh(ctx) {
		t.Fatalf("revive not applied: ids=%v", x.ids)
	}

	// A foreign commit before an own write drops the index.
	env.seedVec(t, "c", "fact", randVec(rng))
	if _, err := mem.IndexWrite(ctx, func() ([]string, error) { return nil, nil }); err != nil {
		t.Fatal(err)
	}
	if x.loaded {
		t.Fatal("foreign write before an own write was not detected")
	}

	// Invalidate forces a reload on the next query.
	_, got := searchBoth(t, env, mem, Query{}, qvec, 10)
	if len(got) != 3 || !x.loaded {
		t.Fatalf("reload: %v", got)
	}
	mem.InvalidateIndex()
	if x.loaded || x.ids != nil {
		t.Fatal("InvalidateIndex did not drop the index")
	}
	if _, got = searchBoth(t, env, mem, Query{}, qvec, 10); len(got) != 3 {
		t.Fatalf("after invalidate: %v", got)
	}
}

func TestDenseIndex_Remove(t *testing.T) {
	x := &vecIndex{dim: 2, pos: map[string]int{}}
	for i, id := range []string{"a", "b", "c"} {
		x.set(id, embed.EncodeVec([]float32{float32(i + 1), 0}))
	}
	x.remove("a") // c moves into a's slot
	if len(x.ids) != 2 || x.ids[0] != "c" || x.pos["c"] != 0 || x.pos["b"] != 1 || len(x.mat) != 4 {
		t.Fatalf("after remove: ids=%v pos=%v mat=%v", x.ids, x.pos, x.mat)
	}
	x.set("b", embed.EncodeVec([]float32{0, 3})) // replace in place, normalized
	if x.mat[2] != 0 || x.mat[3] != 1 {
		t.Fatalf("replace: %v", x.mat)
	}
	x.set("b", []byte{1}) // wrong length removes
	if len(x.ids) != 1 {
		t.Fatalf("wrong-length blob kept: %v", x.ids)
	}
}

func randVecN(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func sameHits(t *testing.T, label string, want, got []hit) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: %d hits, want %d", label, len(got), len(want))
	}
	for i := range want {
		if want[i].ID != got[i].ID || math.Abs(want[i].Score-got[i].Score) > 1e-5 {
			t.Fatalf("%s rank %d: got %v, want %v", label, i, got[i], want[i])
		}
	}
}

// The index is built from the embedder's guessed dimension; the SQLite path
// is the reference for what it must return afterwards.
func TestDenseIndex_DimensionMismatch(t *testing.T) {
	const small, big = 24, 40
	for _, tc := range []struct {
		name  string
		guess int // vecIndex.dim at construction (cfg.Dim)
		rows  map[int]int
		dims  []int // query dimensions, in order
	}{
		{"matching", big, map[int]int{big: 200}, []int{big}},
		{"server dim differs from cfg dim", small, map[int]int{big: 200}, []int{big}},
		{"mixed rows, each query dim", big, map[int]int{big: 150, small: 100}, []int{big, small, big, small}},
		{"mixed rows, guess matches neither", 7, map[int]int{big: 150, small: 100}, []int{small, big}},
		{"no stored row matches", small, map[int]int{big: 20}, []int{small, big}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			env, mem := newIndexEnv(t)
			mem.idx.dim = tc.guess
			rng := rand.New(rand.NewSource(11))
			n := 0
			for dim, count := range tc.rows {
				for i := 0; i < count; i++ {
					env.seedVec(t, "r"+itoa(n), "fact", randVecN(rng, dim))
					n++
				}
			}
			for _, dim := range tc.dims {
				qvec := randVecN(rng, dim)
				want, got := searchBoth(t, env, mem, Query{}, qvec, 10)
				if tc.rows[dim] > 0 && len(want) == 0 {
					t.Fatalf("dim %d: reference returned nothing", dim)
				}
				sameHits(t, "dim "+itoa(dim), want, got)
				if mem.idx.dim != dim {
					t.Fatalf("index dim = %d after a dim-%d query", mem.idx.dim, dim)
				}
			}
		})
	}
}

// Own writes of a vector with another width must not corrupt the matrix,
// and the rows must still agree with the SQLite path afterwards.
func TestDenseIndex_IncrementalAddWrongDim(t *testing.T) {
	ctx := context.Background()
	env, mem := newIndexEnv(t)
	rng := rand.New(rand.NewSource(5))
	for i := 0; i < 10; i++ {
		env.seedVec(t, "a"+itoa(i), "fact", randVec(rng))
	}
	qvec := randVec(rng)
	searchBoth(t, env, mem, Query{}, qvec, 5) // load at idxDim
	x := mem.idx

	write := func(id string, vec []float32) {
		t.Helper()
		if _, err := mem.IndexWrite(ctx, func() ([]string, error) {
			env.seedVec(t, id, "fact", vec)
			return []string{id}, nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	write("short", randVecN(rng, idxDim/2))
	write("long", randVecN(rng, idxDim*2))
	write("good", qvec)
	if len(x.ids) != 11 || len(x.mat) != 11*idxDim {
		t.Fatalf("matrix corrupted: ids=%d mat=%d", len(x.ids), len(x.mat))
	}
	if _, ok := x.pos["short"]; ok {
		t.Fatal("wrong-dim vector was indexed")
	}
	if !x.loaded || !x.fresh(ctx) {
		t.Fatal("index should have stayed loaded and fresh")
	}
	want, got := searchBoth(t, env, mem, Query{}, qvec, 20)
	sameHits(t, "after wrong-dim adds", want, got)
	if got[0].ID != "good" {
		t.Fatalf("top hit %v, want good", got[0])
	}
	// Re-querying at the other widths finds those rows.
	for _, n := range []int{idxDim / 2, idxDim * 2, idxDim} {
		want, got = searchBoth(t, env, mem, Query{}, randVecN(rng, n), 20)
		sameHits(t, "dim "+itoa(n), want, got)
	}
}

func TestDot32_BoundsSafe(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := make([]float32, 3) // no spare capacity
	copy(b, []float32{1, 1, 1})
	if got := dot32(a, b); got != 6 {
		t.Fatalf("dot32(long, short) = %v, want 6", got)
	}
	if got := dot32(b, a); got != 6 {
		t.Fatalf("dot32(short, long) = %v, want 6", got)
	}
	if got := dot32(nil, a); got != 0 {
		t.Fatalf("dot32(nil, a) = %v", got)
	}
}

// Rank must not panic or read across rows if asked with the wrong width.
func TestDenseIndex_RankWrongWidth(t *testing.T) {
	x := &vecIndex{dim: 4, pos: map[string]int{}}
	x.set("a", embed.EncodeVec([]float32{1, 0, 0, 0}))
	hits, err := x.rank(context.Background(), []float32{1, 0}, 5, "1", nil)
	if err != nil || hits != nil {
		t.Fatalf("rank = %v, %v; want nothing", hits, err)
	}
}

// Searches at alternating widths racing own writes must be data-race free
// (run with -race) and still agree with the reference once quiet.
func TestDenseIndex_ConcurrentDimChange(t *testing.T) {
	ctx := context.Background()
	env, mem := newIndexEnv(t)
	rng := rand.New(rand.NewSource(9))
	const small = 16
	for i := 0; i < 100; i++ {
		env.seedVec(t, "b"+itoa(i), "fact", randVec(rng))
		env.seedVec(t, "s"+itoa(i), "fact", randVecN(rng, small))
	}
	where, args := filterSQL(Query{}, "i")
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(g)))
			for i := 0; i < 30; i++ {
				n := idxDim
				if (i+g)%2 == 0 {
					n = small
				}
				if _, err := mem.idx.search(ctx, randVecN(r, n), 5, where, args); err != nil {
					t.Error(err)
					return
				}
			}
		}(g)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			id := "w" + itoa(i)
			vec := randVecN(rand.New(rand.NewSource(int64(i))), []int{idxDim, small}[i%2])
			if _, err := mem.IndexWrite(ctx, func() ([]string, error) {
				now := time.Now().UTC().Format(time.RFC3339)
				if _, err := env.d.Raw.Exec(`INSERT INTO items(id, kind, title, body, provenance, tier, created_at, updated_at)
					VALUES(?, 'fact', 't', '', '{}', 2, ?, ?)`, id, now, now); err != nil {
					return nil, err
				}
				_, err := env.d.Raw.Exec("INSERT INTO vec(item_id, embedding) VALUES(?,?)", id, embed.EncodeVec(vec))
				return []string{id}, err
			}); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	wg.Wait()
	for _, n := range []int{idxDim, small} {
		want, got := searchBoth(t, env, mem, Query{}, randVecN(rng, n), 10)
		sameHits(t, "dim "+itoa(n), want, got)
	}
}
