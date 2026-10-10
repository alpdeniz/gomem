# Changelog

## 0.1.0 — 2026-10-10

### Added
- Module path `github.com/alpdeniz/gomem`, `gomem.Version`, and a `pyproject.toml` for the Python client (`send_dimensions`, `embed_length_threshold`).
- `Item.Supersedes` names the item a new one replaces; `Item.SupersededBy` is now read-only output (it was inverted on read).
- `ReflectReport.ItemsPurged`.
- `RememberMany`: one embed batch and one transaction, all-or-nothing (also in the Python bridge).
- `Config.DenseIndex` (`"sqlite"` default, `"memory"`), `Config.FusionMode` (`"legacy"` default, `"minmax"`, `"rrf"`), and `Embed.SendDimensions`.
- The OpenAI and llama.cpp embedders retry on HTTP 429, honouring `Retry-After`.
- `gomem.Embedder`, the interface `Config.Embedder` accepts, is now exported (it was only reachable as an unnameable internal type).

### Fixed
- The keyword-only fallback when no llama-server is running no longer costs ~3.5 s per `Remember`/`Recall`. A refused connection is retried once after 100 ms (not three times with 0.25/1/2.25 s backoff), and after a connection-level failure, 5xx or 501 the embedder (llama.cpp and OpenAI-compatible) is skipped for 30 s: calls fail immediately, then a single probe re-tests it and a success resets it. HTTP 501 and 505 are no longer retried, and caller cancellation never starts a cool-down.
- Corruption recovery no longer deletes the `-wal` file: it is moved (with `-shm`) next to the `.corrupt-<stamp>` backup, so committed-but-uncheckpointed data is preserved and the backup opens as a complete database. Parsing a stored timestamp no longer logs through the global `slog` default.
- The in-memory dense index (`DenseIndex: "memory"`) no longer returns nothing, garbage scores, or panics when the embedder's real dimension differs from the configured one or stored vectors have mixed dimensions; it now reloads at the query's dimension and skips rows of other widths, matching the SQLite path.
- Re-`Remember`ing unchanged content now backfills a vector that was missing because the original embedding failed (previously the item stayed keyword-only forever).
- `RecallQuery.RecencyHalfLifeDays` was accepted but ignored; it now sets the recency half-life of that query's salience term.
- The Python bridge now returns a usable handle together with the error when `New` recovers from a corrupt file (matching Go); it used to leak the store and fail the open.

### Changed
- Internal refactor: provenance, usage history, timestamps and the salience scalar live in one `internal/model` package (replacing `timeutil` and the triplicated `ProvDoc`); the two HTTP embedders share one transport; recall's two filter lookups, the dense/sparse hit types and the reflect/remember SQL sequences are unified. A malformed stored provenance now reads as full-confidence `run` from `Get` too, as it already did in recall and reflect.
- `Remember` no longer writes document vectors into the 10k-entry query LRU, where bulk writes evicted the goal vectors `Recall` reuses. Documents use a separate small (1024-entry) LRU plus the shared disk cache, so repeated identical content is still embedded once.
- **API change (decision D1):** `Item.Tier` is now `*int`. nil means unspecified: a new item starts hot (2) instead of cold (0), and re-`Remember`ing an existing item keeps its stored tier instead of resetting it to 0. A non-nil value is clamped to 0..2 and overwrites, so `Tier: &zero` stores a cold item. `Get` always returns it non-nil, so Get-then-Remember preserves the tier. The Python `tier` argument defaults to `None` (omitted on the wire) and the bridge maps an absent `"tier"` to nil, so Go and Python now agree. Existing rows are unchanged. `gomem.Tier(n)` builds the pointer inline.
- **API change (decision D2):** `Provenance.Confidence` is now `*float64` (`gomem.Confidence(x)` builds it). nil means unspecified and is stored as 1; an explicit `0` is now kept, so an item can be marked untrusted and is excluded by any `MinConfidence > 0`. Values are clamped to [0,1]; NaN is `ErrInvalid`. `Get` and `Hit` always return it non-nil. Python/bridge: an absent `"confidence"` is 1 and `0` is honoured (the Python default stays 0.8). Previously any value <= 0 became 1.
- Age-based eviction in `Reflect` now depends only on idle time: an item whose last access (creation, if never accessed) is more than `EvictionAgeDays` ago is evicted, regardless of confidence or lifetime access count; a `Touch` resets the clock. It used to also require a salience below a hidden 0.25, so eviction started near 100 days at confidence 1, started at 90 days for confidence 0.8, and never happened for frequently accessed items. Salience now only orders eviction over `MaxItems`. Tier demotion and age eviction run as SQL, and `Reflect` loads rows only when the live count exceeds `MaxItems` (about 2x faster on 30k items).
- `Recall` embeds the goal before taking the database lock (as `Remember` already does), so a slow embedder no longer stalls writers, and through them new readers.
- `Recall` runs the dense and sparse legs concurrently: latency is the slower leg rather than the sum. A cancelled context now returns its error instead of an empty result.
- With the default (`legacy`) fusion, a hit found only by the keyword leg is scored with its real dense similarity instead of 0, so exact identifier matches are no longer penalized by about 0.25. `minmax` and `rrf` are unchanged.
- `Reflect` triggers now differ: `run_end` no longer purges tombstones; `periodic` purges (and vacuums when over 25% of pages are free); `budget_pressure` purges and always vacuums. `Stats.LastVacuumAt` is now set. VACUUM temporarily needs about the database's size in spare disk.
- A superseded item becomes visible again when its superseder is deleted (it used to stay hidden until the superseder was purged 7 days later).
- The embedding cache key now includes the provider, model and dimension, so existing on-disk cache entries are orphaned and re-embedded on first use. Delete the old cache directory to reclaim the space.
- The offline embedder (and keyword tokenization, dedup) is Unicode-aware, so offline vectors change for non-ASCII text; re-`Remember` such items to refresh them.
- `go` directive is `1.26.0`.
- `Config.Dim` defaults to 512 (was 256). Zero-valued `Config` fields now fall back to exactly the `DefaultConfig()` values, so a zero `DefaultTopK` means 20 (it was 10).
- An unknown `Embed.Provider` now makes `New` fail with `ErrInvalid` instead of silently using llama.cpp. Matching is case-insensitive; `llamacpp`/`llama`, `hash` and `api` are accepted aliases.
- The Python `Memory(...)` tuning arguments default to `None` and are omitted, so the Go defaults apply: a new Python store is now 512-dim (it was 256) and its `top_k` default is 20 (it was 10). `provider` still defaults to `offline`.
- `Remember`/`RememberMany` no longer scan the whole FTS table on every write: schema migration 3 adds `items.fts_rowid` (backfilled from `fts`) and replaces/purges FTS rows by rowid instead of the unindexed `item_id` (9 ms → ~0.1 ms at 30K items). Migration 3 also drops the unused `idx_items_created` index; the first open of an existing database runs a one-time backfill.
- Recall's keyword leg no longer costs time proportional to the number of matching rows for natural-language goals: common English stopwords are dropped from the FTS query (a query made only of stopwords keeps them; indexing is unchanged), and FTS ranks the matches without a join, with the live/tag/kind/tier/confidence filters applied only to the best candidates (more are fetched only when too few pass). Filtered results are unchanged. At 30K items a goal like "how do I fix the slow database" takes about 17 ms instead of 85 ms in the keyword leg.

### Notes
- `ReflectReport.ItemsPromoted`/`ItemsCompacted` and the errors `ErrDuplicate`/`ErrCapacityExceeded` are reserved and never produced.
