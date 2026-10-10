// Package remember implements atomic item persistence. Remember is
// idempotent for unchanged content: re-remembering an identical item is a
// no-op, except that a vector missing because an earlier embed failed is
// backfilled (only the vec row is written). It covers
// ULID assignment, input validation/normalization, embedding (with
// graceful offline degradation), and transactional upsert across the
// items, vec, fts, and tags tables plus supersession links.
package remember

import (
	"crypto/rand"
	"encoding/binary"
	"sync"
	"time"
)

// crockfordBase32 is the ULID alphabet (no I, L, O, U).
const crockfordBase32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// Monotonicity state. Within a single millisecond the 80-bit random
// tail is bumped so that successive ULIDs sort after one another,
// which makes ids orderable by creation time. Guarded by ulidMu.
var (
	ulidMu       sync.Mutex
	ulidPrevMS   uint64
	ulidPrevTail [10]byte
)

// NewULID generates a 26-character Crockford-base32 ULID: 48 bits of
// millisecond timestamp followed by 80 bits of entropy. Successive
// calls within the same millisecond yield monotonically increasing
// (lexicographically ordered) ids.
func NewULID() string {
	ulidMu.Lock()
	defer ulidMu.Unlock()

	ms := uint64(time.Now().UnixMilli())
	var b [16]byte
	// Place the 48-bit timestamp in the first six bytes (big-endian).
	binary.BigEndian.PutUint64(b[0:8], ms<<16)

	if ms == ulidPrevMS {
		// Same tick: reuse the previous tail and bump it so this id
		// is guaranteed to sort after the last one issued this ms.
		copy(b[6:], ulidPrevTail[:])
		incrementBigEndian(b[6:])
		copy(ulidPrevTail[:], b[6:])
	} else {
		// New tick (or clock skew): draw a fresh random tail.
		_, _ = rand.Read(b[6:])
		ulidPrevMS = ms
		copy(ulidPrevTail[:], b[6:])
	}
	return encodeULID(b)
}

// incrementBigEndian adds one to the big-endian byte slice in place,
// propagating carries from the least-significant byte.
func incrementBigEndian(b []byte) {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] < 0xFF {
			b[i]++
			return
		}
		b[i] = 0
	}
	// 80-bit overflow wraps to zero (practically unreachable).
}

// encodeULID renders a 128-bit big-endian value as 26 base32 characters:
// 5 bits each from the least significant end, the first character holding
// the remaining 3.
func encodeULID(b [16]byte) string {
	hi, lo := binary.BigEndian.Uint64(b[:8]), binary.BigEndian.Uint64(b[8:])
	var out [26]byte
	for i := range out {
		shift := uint(5 * (25 - i))
		var v uint64
		switch {
		case shift >= 64:
			v = hi >> (shift - 64)
		case shift > 59:
			v = lo>>shift | hi<<(64-shift)
		default:
			v = lo >> shift
		}
		out[i] = crockfordBase32[v&31]
	}
	return string(out[:])
}
