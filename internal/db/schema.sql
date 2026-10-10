-- gomem slim schema (v1)
-- Deliberately omits access_log / srs_schedule: access frequency and
-- recency live as counters/fields on items and feed a computed salience.

-- Bookkeeping metadata (schema version, last vacuum, ...).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Core memory items.
CREATE TABLE IF NOT EXISTS items (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    title         TEXT NOT NULL,
    body          TEXT NOT NULL DEFAULT '',
    payload       TEXT,
    provenance    TEXT NOT NULL,
    tier          INTEGER NOT NULL DEFAULT 2,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    last_access   TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT REFERENCES items(id),
    deleted_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_items_kind    ON items(kind);
CREATE INDEX IF NOT EXISTS idx_items_tier    ON items(tier);
CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_items_live    ON items(deleted_at) WHERE deleted_at IS NULL;

-- Dense vectors. Plain BLOB table (little-endian float32 array); flat
-- brute-force KNN is computed in Go to stay pure-Go / CGO-free.
CREATE TABLE IF NOT EXISTS vec (
    item_id   TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL
);

-- Sparse keyword index. FTS5 with the unicode61 tokenizer (NOT porter,
-- which would stem apart code identifiers). item_id is UNINDEXED so it
-- is stored for joins but not tokenized.
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
    item_id UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
);

-- Free-form tags (AND-filtered at recall time).
CREATE TABLE IF NOT EXISTS tags (
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    tag       TEXT NOT NULL,
    PRIMARY KEY (item_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
