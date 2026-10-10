package model

import (
	"testing"
	"time"
)

func TestParseTS(t *testing.T) {
	now := time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC)
	created := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	updated := time.Date(2026, 2, 1, 0, 0, 0, 0, time.UTC)
	for _, tc := range []struct {
		name, s, fallback string
		want              time.Time
	}{
		{"rfc3339", "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", created},
		{"nano", "2026-01-01T00:00:00.123456789Z", "", created.Add(123456789)},
		{"fallback", "garbage", "2026-02-01T00:00:00Z", updated},
		{"now", "garbage", "also garbage", now},
		{"empty", "", "", now},
	} {
		if got := ParseTS(tc.s, tc.fallback, now); !got.Equal(tc.want) {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}
