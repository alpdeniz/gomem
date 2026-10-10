package gomem

import "time"

// Kind categorizes a memory item. It is intentionally a free-form
// string so callers can invent their own categories; the constants
// below are conventional suggestions, not an enforced enum.
type Kind string

// Conventional Kind values.
const (
	KindLesson   Kind = "lesson"
	KindSkill    Kind = "skill"
	KindFact     Kind = "fact"
	KindTemplate Kind = "template"
)

// Provenance records where an item originated and how trustworthy it
// is. It is persisted as a JSON document on the item.
type Provenance struct {
	Source   string `json:"source"`
	RunID    string `json:"run_id,omitempty"`
	NodeRole string `json:"node_role,omitempty"`
	// Confidence is the item's trust in [0,1]. On input nil means
	// unspecified and is stored as 1; an explicit value, including 0
	// ("untrusted"), is kept (clamped to [0,1]; NaN is ErrInvalid). Output
	// (Get, Hit) is always non-nil.
	Confidence   *float64  `json:"confidence,omitempty"`
	EvidenceRefs []string  `json:"evidence_refs,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
}

// Tier returns a pointer to n, for setting Item.Tier inline:
// gomem.Item{Tier: gomem.Tier(0)} stores a cold item.
func Tier(n int) *int { return &n }

// Confidence returns a pointer to c, for setting Provenance.Confidence
// inline: gomem.Provenance{Confidence: gomem.Confidence(0.9)}.
func Confidence(c float64) *float64 { return &c }

// Item is the input to Remember. Kind and Title are required. Defaults
// applied on input:
//   - ID: a new ULID if empty.
//   - Provenance.Source: "run" if empty; CreatedAt: now if zero.
//   - Provenance.Confidence: nil becomes 1. An explicit value (0 included,
//     which marks the item untrusted) is clamped to [0,1]; NaN is
//     ErrInvalid.
//   - Tier: nil means "unspecified": a new item starts hot (2) and
//     updating an existing item keeps its stored tier. A non-nil value is
//     clamped to 0..2 (0 cold, 1 warm, 2 hot) and overwrites the stored
//     one. Get always returns it non-nil, so a Get-then-Remember round
//     trip preserves the tier. Reflect only ever lowers it.
//   - Tags: lowercased, trimmed, de-duplicated.
type Item struct {
	ID         string
	Kind       Kind
	Title      string
	Body       string
	Payload    any
	Provenance Provenance
	Tier       *int
	Tags       []string

	// Supersedes is the ID of an item this one replaces. On Get it is the
	// most recent live item that points at this one (if several do).
	Supersedes string
	// SupersededBy is output only: the ID of the live item that
	// supersedes this one. Set by the store; ignored on input when equal
	// to the stored value, rejected otherwise.
	SupersededBy string
}

// RecallQuery describes a hybrid retrieval request.
type RecallQuery struct {
	// Goal is the free-form semantic anchor, embedded for the dense leg;
	// it also feeds the keyword leg, which drops English stopwords. If
	// the embedder is unavailable, recall is keyword-only.
	Goal string
	// SubGoal adds keyword terms (e.g. named entities) to the sparse leg
	// only; the dense leg stays anchored on Goal.
	SubGoal string

	// Filters (all applied with AND semantics).
	Kinds         []Kind   // nil/empty = all kinds
	MinTier       int      // include items with tier >= MinTier; 0 (default) = no filter
	Tags          []string // every tag must be present
	MinConfidence float64  // exclude items below this provenance confidence

	// Ranking / shaping.
	TopK                int     // default Config.DefaultTopK
	Budget              int     // soft cap on total tokens across results
	RecencyHalfLifeDays float64 // default Config.RecencyHalfLifeDays

	// Options.
	IncludeSuperseded     bool
	DedupJaccardThreshold float64 // default Config.DedupJaccardThreshold
}

// Hit is a single recall result.
type Hit struct {
	ID         string
	Kind       Kind
	Title      string
	Body       string
	Score      float64
	Provenance Provenance
	Tier       int
	Tokens     int
}

// ReflectTrigger names why a housekeeping pass is being requested. Every
// trigger ages items (tier decay and eviction of items idle longer than
// Config.EvictionAgeDays, both based on time since last access), evicts
// the lowest-salience items over capacity, and checkpoints the WAL.
// TriggerPeriodic also purges old tombstones, and vacuums when over a
// quarter of the file is free pages. TriggerBudgetPressure purges and
// always vacuums; VACUUM temporarily needs about the database's size in
// extra disk space.
type ReflectTrigger string

// Reflect trigger values.
const (
	TriggerRunEnd         ReflectTrigger = "run_end"
	TriggerPeriodic       ReflectTrigger = "periodic"
	TriggerBudgetPressure ReflectTrigger = "budget_pressure"
)

// ReflectReport summarizes what a housekeeping pass did.
type ReflectReport struct {
	ItemsDecayed int
	ItemsEvicted int
	ItemsPurged  int
	DurationMs   int
}

// Stats reports cardinality, storage, and health metrics.
type Stats struct {
	TotalItems   int
	ByKind       map[Kind]int
	ByTier       map[int]int
	DBSizeBytes  int64
	WALSizeBytes int64
	LastVacuumAt *time.Time
}
