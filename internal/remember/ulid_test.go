package remember

import (
	"strings"
	"testing"
)

func TestNewULID_Format(t *testing.T) {
	id := NewULID()
	if len(id) != 26 {
		t.Fatalf("expected 26 chars, got %d (%q)", len(id), id)
	}
	for i, r := range id {
		if !strings.ContainsRune(crockfordBase32, r) {
			t.Fatalf("invalid char %q at %d in %q", r, i, id)
		}
	}
}

func TestNewULID_Uniqueness(t *testing.T) {
	const n = 1000
	seen := make(map[string]bool, n)
	for i := 0; i < n; i++ {
		id := NewULID()
		if seen[id] {
			t.Fatalf("duplicate ULID generated: %q", id)
		}
		seen[id] = true
	}
}

func TestNewULID_Monotonic(t *testing.T) {
	first := NewULID()
	second := NewULID()
	// Later ULIDs sort after earlier ones (timestamp dominates).
	if !(second > first) {
		t.Fatalf("expected %q > %q (monotonic)", second, first)
	}
}
