package embed

import (
	"fmt"
	"testing"
)

func TestCache_MemoryRoundTrip(t *testing.T) {
	c := NewCache("", "")
	vec := []float32{0.1, 0.2, 0.3}
	c.Put("hello world", vec)

	got, ok := c.Get("hello world")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if len(got) != 3 || got[0] != 0.1 || got[1] != 0.2 || got[2] != 0.3 {
		t.Fatalf("unexpected vector: %v", got)
	}

	if _, ok := c.Get("never stored"); ok {
		t.Fatal("expected cache miss for unseen content")
	}
}

func TestCache_KeyStability(t *testing.T) {
	c := NewCache("", "")
	k1 := c.Key("stable content")
	k2 := c.Key("stable content")
	if k1 != k2 {
		t.Fatalf("key not stable: %q vs %q", k1, k2)
	}
	if k1 == c.Key("different content") {
		t.Fatal("distinct content produced identical keys")
	}
}

func TestCache_DiskPersistence(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(dir, "")
	vec := []float32{1.1, 2.2, 3.3, 4.4}
	c.Put("persist me across instances", vec)

	// A fresh cache over the same dir should recover the vector.
	c2 := NewCache(dir, "")
	got, ok := c2.Get("persist me across instances")
	if !ok {
		t.Fatal("expected disk-backed hit on a fresh cache instance")
	}
	if len(got) != 4 || got[0] != 1.1 || got[3] != 4.4 {
		t.Fatalf("disk round-trip mismatch: %v", got)
	}
}

func TestCache_LRU_Eviction(t *testing.T) {
	c := NewCache("", "")
	c.query.max = 2 // shrink for the test
	for i, content := range []string{"one", "two", "three"} {
		c.Put(content, []float32{float32(i)})
	}
	// With capacity 2, the earliest entry should have been evicted.
	if _, ok := c.Get("one"); ok {
		t.Fatal("expected the oldest entry to be evicted under LRU pressure")
	}
	if _, ok := c.Get("three"); !ok {
		t.Fatal("expected the newest entry to survive")
	}
}

func TestCache_TouchedEntryIsEvictable(t *testing.T) {
	c := NewCache("", "")
	c.query.max = 3
	c.Put("a", []float32{1})
	c.Get("a")
	for i := 0; i < 10; i++ {
		c.Put(fmt.Sprintf("k%d", i), []float32{1})
	}
	if len(c.query.m) > 3 || c.query.order.Len() > 3 {
		t.Fatalf("cache grew past capacity: map=%d list=%d", len(c.query.m), c.query.order.Len())
	}
	if _, ok := c.Get("a"); ok {
		t.Fatal("expected touched entry to be evicted eventually")
	}
}

func TestCache_GetPromotes(t *testing.T) {
	c := NewCache("", "")
	c.query.max = 2
	c.Put("a", []float32{1})
	c.Put("b", []float32{2})
	c.Get("a")
	c.Put("c", []float32{3})
	if _, ok := c.Get("a"); !ok {
		t.Fatal("expected recently read entry a to survive")
	}
	if _, ok := c.Get("b"); ok {
		t.Fatal("expected least recently used entry b to be evicted")
	}
}

func TestCache_NamespaceIsolation(t *testing.T) {
	dir := t.TempDir()
	c1 := NewCache(dir, "openai|m1|256")
	c2 := NewCache(dir, "openai|m2|256")
	if c1.Key("same text") == c2.Key("same text") {
		t.Fatal("different namespaces produced the same key")
	}
	c1.Put("same text", []float32{1, 2})
	if _, ok := c2.Get("same text"); ok {
		t.Fatal("vector leaked across namespaces")
	}
}

func TestCache_DocumentPathDoesNotEvictQueries(t *testing.T) {
	c := NewCache("", "")
	c.query.max = 4
	c.doc.max = 2
	for i := 0; i < 4; i++ {
		c.Put(fmt.Sprintf("q%d", i), []float32{float32(i)})
	}
	for i := 0; i < 50; i++ {
		doc := fmt.Sprintf("doc%d", i)
		c.PutDocument(doc, []float32{1})
		c.GetDocument(doc)
	}
	for i := 0; i < 4; i++ {
		if _, ok := c.Get(fmt.Sprintf("q%d", i)); !ok {
			t.Fatalf("query vector q%d evicted by document traffic", i)
		}
	}
	if len(c.doc.m) > 2 || c.doc.order.Len() > 2 {
		t.Fatalf("document LRU exceeded capacity: %d", len(c.doc.m))
	}
	if _, ok := c.GetDocument("doc49"); !ok {
		t.Fatal("expected most recent document to be cached")
	}
	if _, ok := c.GetDocument("doc0"); ok {
		t.Fatal("expected old document to be evicted from the bounded document LRU")
	}
}

func TestCache_DocumentDiskHitDoesNotPromoteToQueryLRU(t *testing.T) {
	dir := t.TempDir()
	c := NewCache(dir, "")
	c.PutDocument("persisted doc", []float32{1, 2})

	c2 := NewCache(dir, "")
	c2.query.max = 2
	c2.Put("q1", []float32{1})
	c2.Put("q2", []float32{2})
	if _, ok := c2.GetDocument("persisted doc"); !ok {
		t.Fatal("expected disk-backed document hit")
	}
	if len(c2.query.m) != 2 {
		t.Fatalf("document disk hit leaked into query LRU: %d entries", len(c2.query.m))
	}
	if _, ok := c2.Get("q1"); !ok {
		t.Fatal("query vector evicted by a document read")
	}
}
