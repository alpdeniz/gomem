package recall

import (
	"context"
	"strings"

	"github.com/alpdeniz/gomem/internal/db"
)

// supersededExpr is true while an item's superseder is still live; once
// the superseder is deleted the replaced item is visible again.
func supersededExpr(alias string) string {
	return "(" + alias + ".superseded_by IS NOT NULL AND EXISTS (" +
		"SELECT 1 FROM items s WHERE s.id = " + alias + ".superseded_by AND s.deleted_at IS NULL))"
}

// byPK is the INDEXED BY clause for looking items up by a list of IDs. The
// planner otherwise prefers scanning the partial index on live rows over
// the IN-list lookups, which at 10^5 items costs ~10 ms per query.
const byPK = " INDEXED BY sqlite_autoindex_items_1"

// placeholders returns n comma-separated "?" markers.
func placeholders(n int) string {
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

// filterSQL builds the WHERE conjunction for the query's filters over the
// items table (or alias), so both legs and the final load agree. q.Tags
// must already be normalized (lowercased, de-duplicated).
func filterSQL(q Query, alias string) (string, []any) {
	conds := []string{alias + ".deleted_at IS NULL"}
	var args []any
	if !q.IncludeSuperseded {
		conds = append(conds, "NOT "+supersededExpr(alias))
	}
	if len(q.Kinds) > 0 {
		conds = append(conds, alias+".kind IN ("+placeholders(len(q.Kinds))+")")
		args = append(args, toArgs(q.Kinds)...)
	}
	if q.MinTier > 0 {
		conds = append(conds, alias+".tier >= ?")
		args = append(args, q.MinTier)
	}
	if len(q.Tags) > 0 {
		conds = append(conds, "(SELECT COUNT(DISTINCT t.tag) FROM tags t WHERE t.item_id = "+alias+
			".id AND t.tag IN ("+placeholders(len(q.Tags))+")) = ?")
		args = append(args, toArgs(q.Tags)...)
		args = append(args, len(q.Tags))
	}
	if q.MinConfidence > 0 {
		// Matches the Go fallbacks: malformed JSON counts as 1, valid JSON
		// without the field as 0.
		conds = append(conds, "(CASE WHEN json_valid("+alias+".provenance) "+
			"THEN COALESCE(json_extract("+alias+".provenance, '$.confidence'), 0) ELSE 1 END) >= ?")
		args = append(args, q.MinConfidence)
	}
	return strings.Join(conds, " AND "), args
}

// toArgs converts strings (IDs, kinds, tags) to query arguments.
func toArgs(ids []string) []any {
	args := make([]any, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	return args
}

// passing returns the subset of ids whose items pass the filter (where,
// over alias "i").
func passing(ctx context.Context, d *db.DB, ids []string, where string, args []any) (map[string]bool, error) {
	rows, err := d.Raw.QueryContext(ctx,
		"SELECT i.id FROM items i"+byPK+" WHERE i.id IN ("+placeholders(len(ids))+") AND "+where,
		append(toArgs(ids), args...)...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	live := make(map[string]bool, len(ids))
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		live[id] = true
	}
	return live, rows.Err()
}
