package gomem

import (
	"context"
	"errors"
	"net"
	"strings"

	"github.com/alpdeniz/gomem/internal/errs"
)

// Sentinel errors. Callers should classify with errors.Is and the
// Is*/helpers below rather than comparing messages. The identities
// live in internal/errs so internal packages share them without
// importing this package (which would cycle).
var (
	// ErrNotFound indicates the referenced item does not exist.
	ErrNotFound = errs.ErrNotFound
	// ErrDuplicate is reserved; no operation returns it yet.
	ErrDuplicate = errs.ErrDuplicate
	// ErrInvalid indicates a failed input validation (an invalid Item or
	// an invalid Config passed to New).
	ErrInvalid = errs.ErrInvalid
	// ErrCapacityExceeded is reserved; MaxItems is a soft cap enforced by
	// Reflect, so no operation returns it yet.
	ErrCapacityExceeded = errs.ErrCapacityExceeded
	// ErrEmbedFailed is reserved; no public operation returns it. An
	// embedding failure never fails Remember, RememberMany or Recall: the
	// item is stored without a vector (backfilled by a later identical
	// Remember) and Recall falls back to keyword-only search.
	ErrEmbedFailed = errs.ErrEmbedFailed
	// ErrCorrupt indicates a structural integrity failure.
	ErrCorrupt = errs.ErrCorrupt
	// ErrClosed indicates an operation on a released handle.
	ErrClosed = errs.ErrClosed
)

// IsNotFound reports whether err denotes a missing item.
func IsNotFound(err error) bool {
	return errors.Is(err, ErrNotFound)
}

// IsTransient reports whether err is worth retrying (network blips,
// transient locks).
func IsTransient(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	var ne net.Error
	if errors.As(err, &ne) {
		return true
	}
	msg := strings.ToLower(err.Error())
	for _, tok := range []string{"busy", "locked", "deadline", "temporary"} {
		if strings.Contains(msg, tok) {
			return true
		}
	}
	return false
}
