# gomem

**0.1.0 — the API may still move.**

An embeddable, pure-Go memory substrate for autonomous agents. `gomem` gives an agent process durable, self-managing long-term memory with an embedded SQLite store (no database daemon) and no CGO — everything lives in a single portable SQLite file. The default embedder talks to a local llama-server over HTTP; for a fully zero-network setup use the `offline` embedder (`cfg.Embed.Provider = "offline"`).

## Highlights

- **Hybrid recall** — fuses dense-vector (semantic) and sparse-keyword (exact-match) legs, blended with a salience signal, so you get both meaning and precision.
- **Salience** — a weighted sum of confidence, access frequency, and recency (computed on demand, not stored) orders capacity eviction and feeds the recall score. Aging is separate and depends only on time since last access: `Reflect` demotes idle items through the tiers and evicts those idle longer than `EvictionAgeDays`. `Touch` resets that idle clock (and raises salience) but never restores a demoted tier.
- **Provenance tracking** — every item records where it came from and how trustworthy it is.
- **Zero-ops & portable** — a single `.db` file (WAL mode) that you can copy to move an agent's entire memory. Measured recall latency (Apple Silicon, mock embedder): ~50 ms at 10K items and ~480 ms at 100K with the default `DenseIndex: "sqlite"`; ~290 ms at 100K with `"memory"`. The dense leg is ~8 ms at 100K, and the rest is the sparse leg on a benchmark corpus where every item matches the query. The keyword leg drops English stopwords and ranks in FTS before filtering, so queries with few matching terms do less work; the figures above are not a guarantee for other corpora.
- **Semantic by default** — recall is driven by a local **llama.cpp** server (true dense semantics, no cloud hop); swap in the zero-network feature-hash embedder or any OpenAI-compatible endpoint with one config field.
- **Normalized legs** — dense and sparse (BM25) scores are mapped onto `[0,1]` before the weighted fusion with salience (see `FusionMode` under [Configuration](#configuration)), and recall logs each leg's realized contribution at debug level so you can tune `AlphaDense`/`BetaSparse`/`GammaSalience`.

## Requirements

- **Go 1.26+** (go.mod targets 1.26.0)
- No CGO required (pure-Go SQLite via `modernc.org/sqlite`)

## Quickstart

```sh
go get github.com/alpdeniz/gomem
```

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/alpdeniz/gomem"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Open (or create) the memory store.
	cfg := gomem.DefaultConfig()
	cfg.Path = "my-agent.db"
	cfg.Embed.Provider = "offline" // llama.cpp is the default; this keeps the snippet zero-server
	mem, err := gomem.New(ctx, cfg)
	if err != nil && !errors.Is(err, gomem.ErrCorrupt) {
		log.Fatalf("open: %v", err)
	}
	defer mem.Close()

	// Remember something.
	id, err := mem.Remember(ctx, gomem.Item{
		Kind:  gomem.KindFact,
		Title: "Composite indexes speed up filtered queries",
		Body:  "Aligning a composite index with WHERE and ORDER BY clauses avoids table scans and sorts.",
		Tags:  []string{"database", "index"},
		Provenance: gomem.Provenance{
			Source:     "run",
			Confidence: gomem.Confidence(0.9),
		},
	})
	if err != nil {
		log.Fatalf("remember: %v", err)
	}
	fmt.Println("stored:", id)

	// Recall the most relevant memories for a goal.
	hits, err := mem.Recall(ctx, gomem.RecallQuery{
		Goal: "speed up slow database queries",
		TopK: 5,
	})
	if err != nil {
		log.Fatalf("recall: %v", err)
	}
	for rank, h := range hits {
		fmt.Printf("#%d score=%.3f [%s] %s\n", rank+1, h.Score, h.Kind, h.Title)
	}
}
```

> **Note:** The snippet sets `offline` so it runs with no server. `DefaultConfig()` uses the **llama.cpp** embedder, so the dense (semantic) leg expects a local llama-server at `cfg.Embed.APIURL` (default `http://127.0.0.1:8080`). When the server is unreachable, the embedder is skipped for a short cool-down (about 30 s) after a failed embed, and `Recall`/`Remember` fall back to keyword-only (FTS) search; the first call after startup or after the cool-down expires may pay a short bounded delay. An item stored without a vector is backfilled by re-`Remember`ing the same content once a server is available. See [Embeddings](#embeddings-llamacpp-by-default).

### Core API

| Method | Purpose |
|--------|---------|
| `New(ctx, cfg)` | Open/create the store and wire all engines. |
| `Remember(ctx, item)` | Persist a new item (or update by ID); returns its ID (yours if set, else a new ULID). Idempotent for unchanged content: re-remembering an identical item is a no-op, except that it backfills a vector missing from an earlier failed embed. |
| `RememberMany(ctx, items)` | Persist several items in one embed batch and one transaction (all-or-nothing); returns their IDs. About 0.5 ms/item at 1k items versus ~3.85 ms/item for sequential `Remember` (mock embedder). |
| `Recall(ctx, query)` | Hybrid retrieval; returns scored hits, best-first. |
| `Get(ctx, id)` | Fetch one item, or `ErrNotFound` if it is absent or deleted. |
| `Delete(ctx, id)` | Soft-delete an item; `Reflect` purges it after 7 days. |
| `Touch(ctx, id)` | Record an access (bumps the access count and last-access time), which raises its salience and resets its idle (last-access) clock for aging; it never restores a demoted tier. |
| `Reflect(ctx, trigger)` | Housekeeping: tier decay and eviction, then WAL checkpoint. An item idle (no access) for longer than `EvictionAgeDays` is evicted regardless of confidence or access count; over `MaxItems`, the lowest computed (unstored) salience is evicted first; `periodic` also purges old tombstones and `budget_pressure` also vacuums (needs about the database size in spare disk). |
| `Stats(ctx)` | Cardinality, storage, and health metrics. |
| `Close()` | Flush the WAL and release the handle (idempotent). |

Conventional `Kind` values: `lesson`, `skill`, `fact`, `template` (free-form strings are allowed).

`Item.Supersedes` names an item the new one replaces; `SupersededBy` is read-only output. `Item.Tier` and `Provenance.Confidence` are pointers so "unspecified" differs from zero: a nil `Tier` starts a new item hot (and keeps the stored tier on update), a nil `Confidence` is stored as 1, and an explicit `gomem.Tier(0)` / `gomem.Confidence(0)` is kept. Use the `gomem.Tier` and `gomem.Confidence` helpers to set them inline.

Sentinel errors are exposed for classification via `errors.Is`: `ErrNotFound`, `ErrInvalid`, `ErrCorrupt`, `ErrClosed`, plus helpers `IsNotFound` and `IsTransient`. `New` returns a usable `Memory` together with an error wrapping `ErrCorrupt` when it found the file corrupt, moved it (and its `-wal`/`-shm` files) aside as a backup, and created a fresh database. `ErrDuplicate`, `ErrEmbedFailed` and `ErrCapacityExceeded` are reserved and never returned today (an embedding failure is not an error: the item is stored keyword-only); `MaxItems` is a soft cap enforced by `Reflect`. Likewise `ReflectReport.ItemsPromoted` and `ItemsCompacted` are always 0.

## Configuration

`DefaultConfig()` gives working defaults; the fields that change recall behaviour:

| Field | Values | Effect |
|-------|--------|--------|
| `DenseIndex` | `"sqlite"` (default), `"memory"` | `sqlite` scans stored vectors on each query. `memory` keeps a copy in RAM (N × dim × 4 bytes) for faster recall at large sizes. |
| `FusionMode` | `"legacy"` (default), `"minmax"`, `"rrf"` | How leg scores reach `[0,1]`: `legacy` maps dense cosine as (cos+1)/2 and min-max scales sparse BM25 over the fetched batch; `minmax` min-max scales both; `rrf` uses reciprocal rank, 1/(60+rank). |
| `Embed.SendDimensions` | `*bool`, nil = true | Whether the `openai` provider sends `dimensions`; set a pointer to `false` for servers that reject it. |
| `EmbedLengthThreshold` | int, default 10 | Minimum rune length of title+body to embed. Shorter items are stored keyword-only; `0` embeds everything. |
| `AlphaDense`, `BetaSparse`, `GammaSalience` | floats | Fusion weights. Not defaulted: they must sum to more than zero (as must `WConfidence`/`WFrequency`/`WRecency`), so start from `DefaultConfig()`; a bare `Config{}` makes `New` fail with an error wrapping `ErrInvalid`. |

## Embeddings (llama.cpp by default)

`gomem`'s dense (semantic) leg is powered by a pluggable embedder. **Out of the box the default is a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server**, so you get *true* semantic recall with no cloud dependency — the model runs on your machine.

### Point it at your llama-server

Start a llama-server with embeddings enabled (`--embeddings`), which exposes the `/embedding` endpoint, then point `gomem` at it:

```sh
# Serve a sentence-embedding GGUF on the default port.
./llama-server -m ./models/all-MiniLM-L6-v2.gguf --embeddings --port 8080
```

Without `--embeddings` the server answers HTTP 501 to `/embedding`; `gomem` treats that as "embeddings disabled", does not retry it, and skips the server for the short cool-down, so recall runs keyword-only.

```go
cfg := gomem.DefaultConfig()
cfg.Path = "my-agent.db"

// llama.cpp is already the default provider; just aim it at your server.
cfg.Embed.APIURL = "http://127.0.0.1:8080" // your llama-server
cfg.Embed.Model  = "all-MiniLM-L6-v2"  // optional, if the server multiplexes
cfg.Embed.APIKey = "sk-..."             // optional; sent as "Authorization: Bearer <key>" if your server/proxy requires auth
cfg.Dim           = 384              // a hint; see below
```

> **Credentials and the server URL come from the environment by default.** When `cfg.Embed.APIKey` or `cfg.Embed.APIURL` is empty, it is read from `GOMEM_EMBED_API_KEY` or `GOMEM_EMBED_API_URL`. An explicit value (programmatically or via the Python bridge) always wins over the environment.

> **`cfg.Dim` is only a hint for llama.cpp.** The embedder learns the real dimension from the server's first response and uses that (logging a warning if it differs from `cfg.Dim`). For the `openai` provider `cfg.Dim` is sent as the `dimensions` request field; set `cfg.Embed.SendDimensions` to a pointer to `false` for compatible servers that reject unknown fields. Changing models or dimensions afterwards invalidates previously stored vectors. Re-`Remember`ing an item with unchanged content does **not** re-embed it (it is a no-op that only backfills a *missing* vector), so to switch models re-`Remember` your items into a fresh database file.

### Other providers

Set `cfg.Embed.Provider` to switch (case-insensitive; an unknown value makes `New` fail with `ErrInvalid`):

| Provider | What it does | When to use |
|----------|----------------|--------------|
| `llama.cpp` (**default**) | Calls a local llama-server's `/embedding` | True semantics, fully local, no cloud |
| `offline` | Deterministic feature-hash vectorizer, zero network | No server available; lexical-grade recall |
| `openai` | OpenAI-compatible `/embeddings` HTTP | Cloud or self-hosted OpenAI-compatible APIs |

For any other backend, implement `gomem.Embedder` (`Embed(ctx, texts) ([][]float32, error)` and `Dim() int`) and set `cfg.Embedder`; it overrides `cfg.Embed`.

With a debug-level `cfg.Logger`, each recall logs its **realized per-leg contribution** (nominal weight × mean leg score, as a percentage). Use it when calibrating `AlphaDense`/`BetaSparse`/`GammaSalience` against what the legs actually contribute, not their nominal values.

## Try the example

A complete end-to-end demo (remember → recall → stats → close) lives in [`examples/basic_recall`](examples/basic_recall):

```sh
go run ./examples/basic_recall
```

It prints the stored items, the ranked recall results for a sample goal, and a stats summary. It uses the `offline` embedder so it runs with no server; set `cfg.Embed.Provider = "llama.cpp"` for true semantics.

`examples/embed_bench` measures `Remember` throughput and latency against a live embedding server (`make bench-embed`, which reads `GOMEM_EMBED_API_URL`/`GOMEM_EMBED_API_KEY` from a local, git-ignored `.env`).

## Python integration

The Python client is a source package (`pip install -e .`) over a `c-shared`
bridge (`go build -buildmode=c-shared`), which requires `CGO_ENABLED=1` and a
C toolchain. The library itself needs no CGO. The bridge in [`python/`](python/)
exposes the full API (`New`, `Remember`, `RememberMany`, `Get`, `Delete`,
`Touch`, `Recall`, `Reflect`, `Stats`, `Close`) as JSON-in/JSON-out entry
points, and [`python/gomem.py`](python/gomem.py) is a ctypes client wrapping
them in a small ergonomic facade.

```sh
pip install -e .                          # optional; or add python/ to PYTHONPATH
sh python/build.sh                        # -> python/libgomem.dylib (macOS) / .so (Linux) / gomem.dll
python3 examples/python_demo/demo.py    # end-to-end demo (offline embedder)
```

```python
from gomem import Memory

mem = Memory("/tmp/agent.db", provider="offline")   # or "llama.cpp" / "openai"
mem.remember(kind="fact", title="...", body="...", tags=["db"])
for h in mem.recall("speed up slow database queries", top_k=5):
    print(h["score"], h["title"])
mem.stats(); mem.close()
```

Notes:

- `Memory(...)` settings left unset fall back to Go's `DefaultConfig()`.
  Three Python defaults differ on purpose: provider `offline` (not
  `llama.cpp`), confidence 0.8 (not 1) and `top_k` 5 on `recall`.
  `tier` is optional everywhere (as `Item.Tier` is in Go): omitted, a
  new item starts hot and an update keeps its stored tier; pass `tier=0`
  to store a cold item.
- The demo uses the zero-network `offline` embedder so it runs anywhere;
  point `provider="llama.cpp"` + `api_url=` at a local llama-server for true
  semantics.
- The library name is per-OS so a build for one platform is never loaded
  on another. Set `GOMEM_LIB=/path/to/lib` to override discovery.
- A custom `gomem.Embedder` is Go-only; from Python, pick one of the
  built-in providers (llama.cpp / openai / offline).
- Opening a store recovered from a corrupt file succeeds (as in Go);
  `Memory.recovered_from_corrupt` is then True.

## Build and test

```sh
make          # format check, vet, build, Go tests, Python tests
make help     # all targets (test-race, bench, python, demo, ...)
```

## Project layout

```
.
├── gomem.go                # public Memory interface + New
├── config.go               # Config, EmbedCfg, DefaultConfig, Embedder
├── types.go                # Item, RecallQuery, Hit, Kind, Provenance, ...
├── convert.go              # public <-> internal conversion, embedder selection
├── errors.go               # sentinel errors + helpers
├── Makefile                # build / test / bench / python targets
├── pyproject.toml          # Python package metadata
├── SECURITY.md             # private vulnerability reporting
├── .github/workflows/ci.yml
├── docs/
│   ├── go-memory-library-plan.md   # original design (with status vs implementation)
│   └── remediation-plan.md         # October 2026 review and its task log (historical)
├── internal/
│   ├── db/                 # SQLite open/migrate/WAL/vacuum
│   ├── embed/              # llama.cpp (default), offline + OpenAI-compatible embedders, cache
│   ├── errs/               # shared sentinel errors
│   ├── model/              # provenance, usage, timestamps and salience shared by the engines
│   ├── recall/             # dense + sparse + salience fusion, dedup
│   ├── remember/           # atomic persist (item + vector + FTS + tags)
│   ├── reflect/            # tier decay, eviction, purge, vacuum
│   ├── stats/              # metrics
│   ├── textutil/           # tokenization and stopwords
│   └── testutil/           # fixtures, mock embedder, assertions
├── examples/basic_recall/  # runnable Go demo
├── examples/embed_bench/   # Remember benchmark against a live embedding server
├── examples/python_demo/   # runnable Python demo (ctypes)
└── python/                 # cgo bridge (libgomem) + Python client
```

## License

MIT
