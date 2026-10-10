package model

import "time"

// Stamp formats t as the RFC3339 UTC timestamp stored in the database.
func Stamp(t time.Time) string { return t.UTC().Format(time.RFC3339) }

// ParseTS parses a stored RFC3339 timestamp. When s is malformed it
// falls back to fallback (typically updated_at) and then to now, so
// recall and reflect agree on the age of a damaged row. It is pure and
// never logs; the library's logger is configured per instance.
func ParseTS(s, fallback string, now time.Time) time.Time {
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t
	}
	if t, err := time.Parse(time.RFC3339, fallback); err == nil {
		return t
	}
	return now
}
