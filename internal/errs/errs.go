// Package errs holds the sentinel errors shared across gomem's
// internal packages. Keeping them in a leaf package (stdlib-only
// imports) lets internals reference the same identities the public
// API exports without importing the root package, which would cycle.
package errs

import "errors"

var (
	// ErrNotFound indicates the referenced item does not exist.
	ErrNotFound = errors.New("gomem: item not found")
	// ErrDuplicate indicates a conflicting duplicate.
	ErrDuplicate = errors.New("gomem: duplicate item")
	// ErrInvalid indicates a failed input validation.
	ErrInvalid = errors.New("gomem: invalid item")
	// ErrCapacityExceeded indicates the item limit was breached.
	ErrCapacityExceeded = errors.New("gomem: capacity exceeded")
	// ErrEmbedFailed indicates the embedding step errored.
	ErrEmbedFailed = errors.New("gomem: embedding failed")
	// ErrCorrupt indicates a structural integrity failure.
	ErrCorrupt = errors.New("gomem: corrupted record")
	// ErrClosed indicates an operation on a released handle.
	ErrClosed = errors.New("gomem: already closed")
)
