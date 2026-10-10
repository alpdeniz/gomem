# gomem

**0.1.0 — the API may still move.**

Long-term memory for agents. One portable SQLite file, hybrid semantic + keyword recall. Use it from **Go** or **Python**.

```sh
go run ./examples/basic_recall          # no server
sh python/build.sh && python3 examples/python_demo/demo.py
```

The Go library is pure Go (no CGO). The Python client is a ctypes wrapper over a shared library you build from this repo — it is **not** a PyPI wheel.

## Install

**Go 1.26+**

```sh
go get github.com/alpdeniz/gomem
```

**Python 3.10+** (Go + a C toolchain required to build the bridge)

```sh
git clone https://github.com/alpdeniz/gomem.git
cd gomem
sh python/build.sh                 # python/libgomem.dylib | .so | gomem.dll
pip install -e .                   # or: export PYTHONPATH="$PWD/python"
```

CI builds and tests the bridge on Linux and macOS; the Windows build (`gomem.dll`) is untested.

`pip install -e .` installs the Python module only. The shared library must already exist; `pip install gomem` from PyPI will not give you a working client. Override discovery with `GOMEM_LIB=/path/to/lib`.

## Use

Both snippets use `offline` so they run with no embedding server. In Go, `DefaultConfig()` talks to a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server instead — see [Embeddings](#embeddings).

### Go

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/alpdeniz/gomem"
)

func main() {
	cfg := gomem.DefaultConfig()
	cfg.Path = "my-agent.db"
	cfg.Embed.Provider = "offline"

	mem, err := gomem.New(context.Background(), cfg)
	if err != nil && !errors.Is(err, gomem.ErrCorrupt) {
		log.Fatal(err)
	}
	defer mem.Close()

	id, err := mem.Remember(context.Background(), gomem.Item{
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
		log.Fatal(err)
	}
	fmt.Println("stored:", id)

	hits, err := mem.Recall(context.Background(), gomem.RecallQuery{
		Goal: "speed up slow database queries",
		TopK: 5,
	})
	if err != nil {
		log.Fatal(err)
	}
	for i, h := range hits {
		fmt.Printf("#%d %.3f %s\n", i+1, h.Score, h.Title)
	}
}
```

### Python

```python
from gomem import Memory

with Memory("/tmp/agent.db", provider="offline") as mem:
    mem.remember(kind="fact", title="Composite indexes speed up filtered queries",
                 body="Aligning a composite index with WHERE and ORDER BY clauses avoids table scans and sorts.",
                 tags=["db"])
    for h in mem.recall("speed up slow database queries"):
        print(h["score"], h["title"])
```

Async twin of every method (`aremember`, `arecall`, …) plus `async with Memory(...)`:

```python
async with Memory("/tmp/agent.db") as mem:
    await mem.aremember(kind="fact", title="...", body="...")
    hits = await mem.arecall("speed up slow database queries")
```

Python defaults that differ from Go: provider `offline` (not `llama.cpp`), confidence `0.8` (not `1`), `recall(..., top_k=5)`. Everything else falls through to Go's `DefaultConfig()`.

## API

| Go | Python | What it does |
|----|--------|----------------|
| `New` | `Memory(path, ...)` | Open or create the store. |
| `Remember` / `RememberMany` | `remember` / `remember_many` | Persist items. Same ID + same content is a no-op (missing vectors are backfilled). |
| `Recall` | `recall` | Hybrid retrieval, best-first. |
| `Get` / `Delete` / `Touch` | `get` / `delete` / `touch` | Fetch, soft-delete, record an access. |
| `Reflect` | `reflect` | Housekeeping: age idle items, evict over capacity, checkpoint. |
| `Stats` / `Close` | `stats` / `close` | Metrics; flush and release (idempotent). |

Kinds are free-form; conventional values are `fact`, `lesson`, `skill`, `template`.

`Item.Supersedes` names the item this one replaces. `SupersededBy` is read-only output.

`Item.Tier` and `Provenance.Confidence` are pointers in Go so “unspecified” is not zero: omit them (or pass `tier=None` in Python) and a new item starts hot / confidence defaults as above. `gomem.Tier(0)` / `tier=0` stores a cold item; `gomem.Confidence(0)` / `confidence=0` stores untrusted. `Get` always returns them set.

Useful errors (`errors.Is`): `ErrNotFound`, `ErrInvalid`, `ErrCorrupt`, `ErrClosed`. `New` still returns a usable store when it wraps `ErrCorrupt` — the old file (plus `-wal`/`-shm`) was moved aside and a fresh DB created. In Python, `mem.recovered_from_corrupt` is then `True`. Embedding failures are not errors: the item is stored keyword-only.

Call `Reflect` periodically (`"run_end"`, `"periodic"`, `"budget_pressure"`). An item idle longer than `EvictionAgeDays` (default 90) is evicted; over `MaxItems`, lowest salience goes first. `Touch` resets that idle clock. `Delete` is a tombstone; `periodic` purges it after 7 days.

## Embeddings

| Provider | Use when |
|----------|----------|
| `offline` | No server. Lexical-grade vectors (feature hash). |
| `llama.cpp` | **Go default.** Local llama-server, true semantics. |
| `openai` | Any OpenAI-compatible `/embeddings` URL. |

```sh
./llama-server -m ./models/all-MiniLM-L6-v2.gguf --embeddings --port 8080
```

```go
cfg := gomem.DefaultConfig()          // provider llama.cpp
cfg.Embed.APIURL = "http://127.0.0.1:8080"
```

```python
Memory("agent.db", provider="llama.cpp", api_url="http://127.0.0.1:8080")
```

Empty `APIURL` / `APIKey` are read from `GOMEM_EMBED_API_URL` / `GOMEM_EMBED_API_KEY`. If the server is down, `Remember`/`Recall` fall back to keyword search for ~30s, then probe again. Re-`Remember` the same content to backfill a missing vector once the server is up.

`cfg.Dim` is a hint for llama.cpp (the first response wins) and the `dimensions` field for `openai`. Changing model or dimension invalidates stored vectors — use a new `.db` file. A custom `gomem.Embedder` is Go-only; Python uses the three built-in providers.

Start from `DefaultConfig()` (or omit Python kwargs). A bare `Config{}` fails: fusion and salience weights must sum to more than zero. The knobs that change recall: `DenseIndex` (`"sqlite"` or `"memory"`), `FusionMode` (`"legacy"`, `"minmax"`, `"rrf"`), and `AlphaDense` / `BetaSparse` / `GammaSalience`.

## Develop

```sh
make          # fmt, vet, build, Go + Python tests
make help     # test-race, demo, python-smoke, bench, …
make demo     # go run ./examples/basic_recall
```

[`examples/embed_bench`](examples/embed_bench) measures `Remember` against a live server (`make bench-embed`; reads a git-ignored `.env`).

Public surface: [`gomem.go`](gomem.go), [`config.go`](config.go), [`types.go`](types.go), [`errors.go`](errors.go). Python client: [`python/gomem.py`](python/gomem.py). Demos: [`examples/basic_recall`](examples/basic_recall), [`examples/python_demo`](examples/python_demo).

## License

MIT. Vulnerabilities: [SECURITY.md](SECURITY.md).
