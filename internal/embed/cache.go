package embed

import (
	"container/list"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"sync"
)

// Cache memoizes embeddings by the SHA-256 of their namespace and source
// content, keeping an in-memory LRU and (optionally) a durable on-disk
// store so identical content is embedded once per embedder configuration.
//
// Query vectors (Get/Put) and document vectors (GetDocument/PutDocument)
// use separate in-memory LRUs: documents are already stored durably in the
// vec table, so a bulk write must never evict the query vectors Recall
// reuses. The disk store is shared.
type Cache struct {
	mu        sync.Mutex
	dir       string
	namespace string
	query     *lru
	doc       *lru // small: the write path
}

// lru is a bounded least-recently-used map of vectors; not synchronized.
type lru struct {
	max   int
	m     map[string]*list.Element // key -> element holding *cacheEntry
	order *list.List               // front = most recently used
}

type cacheEntry struct {
	key string
	vec []float32
}

func newLRU(max int) *lru { return &lru{max: max, m: map[string]*list.Element{}, order: list.New()} }

func (l *lru) get(key string) ([]float32, bool) {
	el, ok := l.m[key]
	if !ok {
		return nil, false
	}
	l.order.MoveToFront(el)
	return el.Value.(*cacheEntry).vec, true
}

func (l *lru) put(key string, vec []float32) {
	if el, ok := l.m[key]; ok {
		el.Value.(*cacheEntry).vec = vec
		l.order.MoveToFront(el)
		return
	}
	l.m[key] = l.order.PushFront(&cacheEntry{key: key, vec: vec})
	for l.order.Len() > l.max {
		back := l.order.Back()
		l.order.Remove(back)
		delete(l.m, back.Value.(*cacheEntry).key)
	}
}

// NewCache builds a cache. An empty dir disables disk persistence. The
// namespace (e.g. provider|model|dim) scopes keys so vectors from a
// different embedder configuration are never returned.
func NewCache(dir, namespace string) *Cache {
	return &Cache{dir: dir, namespace: namespace, query: newLRU(10000), doc: newLRU(1024)}
}

// Key derives the stable hash of the namespace and content.
func (c *Cache) Key(content string) string {
	sum := sha256.Sum256([]byte(c.namespace + "\x00" + content))
	return hex.EncodeToString(sum[:])
}

// Get returns a cached vector for the content, if present.
func (c *Cache) Get(content string) ([]float32, bool) { return c.lookup(c.query, content) }

// GetDocument is Get for the document path: it consults the small document
// LRU and the disk store, and never touches or promotes into the query LRU.
func (c *Cache) GetDocument(content string) ([]float32, bool) { return c.lookup(c.doc, content) }

// Put stores a vector for the content.
func (c *Cache) Put(content string, vec []float32) { c.store(c.query, content, vec) }

// PutDocument is Put for the document path: the vector goes to the small
// document LRU and the disk store, leaving the query LRU untouched.
func (c *Cache) PutDocument(content string, vec []float32) { c.store(c.doc, content, vec) }

func (c *Cache) lookup(l *lru, content string) ([]float32, bool) {
	key := c.Key(content)
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := l.get(key); ok {
		return v, true
	}
	if c.dir != "" {
		if v, ok := c.readDisk(key); ok {
			l.put(key, v)
			return v, true
		}
	}
	return nil, false
}

func (c *Cache) store(l *lru, content string, vec []float32) {
	key := c.Key(content)
	c.mu.Lock()
	l.put(key, vec)
	c.mu.Unlock()
	if c.dir != "" {
		c.writeDisk(key, vec)
	}
}

// The disk format is a little-endian uint32 length followed by the vector
// as EncodeVec writes it.

func (c *Cache) diskPath(key string) string {
	return filepath.Join(c.dir, key+".vec")
}

func (c *Cache) writeDisk(key string, vec []float32) {
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return
	}
	buf := binary.LittleEndian.AppendUint32(nil, uint32(len(vec)))
	_ = os.WriteFile(c.diskPath(key), append(buf, EncodeVec(vec)...), 0o600)
}

func (c *Cache) readDisk(key string) ([]float32, bool) {
	buf, err := os.ReadFile(c.diskPath(key))
	if err != nil || len(buf) < 4 || 4+4*int(binary.LittleEndian.Uint32(buf)) != len(buf) {
		return nil, false
	}
	return DecodeVec(buf[4:]), true
}
