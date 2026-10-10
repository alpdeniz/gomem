// Package model holds what remember, recall and reflect must agree on so
// their rules cannot drift apart: the persisted provenance document, an
// item's usage history, timestamps, and the salience scalar computed from
// them.
package model

import (
	"database/sql"
	"encoding/json"
	"slices"
	"time"
)

// ProvDoc is the persisted provenance document.
type ProvDoc struct {
	Source       string    `json:"source"`
	RunID        string    `json:"run_id,omitempty"`
	NodeRole     string    `json:"node_role,omitempty"`
	Confidence   float64   `json:"confidence"`
	EvidenceRefs []string  `json:"evidence_refs,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

// ParseProv decodes a stored provenance document. A malformed one reads as
// a full-confidence "run" item, so the same corruption neither inflates
// recall scores nor accelerates decay.
func ParseProv(b []byte) ProvDoc {
	var p ProvDoc
	if err := json.Unmarshal(b, &p); err != nil {
		return ProvDoc{Source: "run", Confidence: 1}
	}
	return p
}

// Equal reports whether two documents hold the same values.
func (p ProvDoc) Equal(o ProvDoc) bool {
	return p.Source == o.Source && p.RunID == o.RunID && p.NodeRole == o.NodeRole &&
		p.Confidence == o.Confidence && p.CreatedAt.Equal(o.CreatedAt) &&
		slices.Equal(p.EvidenceRefs, o.EvidenceRefs)
}

// Usage is the history an item's salience and idle time derive from.
type Usage struct {
	Confidence  float64
	AccessCount int
	LastAccess  *time.Time // nil: never accessed
	CreatedAt   time.Time
}

// NewUsage builds a Usage from stored columns. A damaged created_at falls
// back to updated_at, then to now (see ParseTS); a damaged last_access
// counts as never accessed.
func NewUsage(confidence float64, accessCount int, createdAt, updatedAt string, lastAccess sql.NullString, now time.Time) Usage {
	u := Usage{Confidence: confidence, AccessCount: accessCount, CreatedAt: ParseTS(createdAt, updatedAt, now)}
	if lastAccess.Valid {
		if t, err := time.Parse(time.RFC3339, lastAccess.String); err == nil {
			u.LastAccess = &t
		}
	}
	return u
}

// Ref is the idle clock's start: the last access, else the creation.
func (u Usage) Ref() time.Time {
	if u.LastAccess != nil {
		return *u.LastAccess
	}
	return u.CreatedAt
}
