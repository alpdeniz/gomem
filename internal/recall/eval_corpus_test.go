package recall

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/alpdeniz/gomem/internal/embed"
)

// The extended evaluation corpus is the hand-written items from eval_test.go
// plus harder material: exact identifiers (error codes, function names),
// natural-language queries full of stopwords, paraphrases with little word
// overlap, near-miss distractors, and several hundred seeded filler items
// that share generic vocabulary with every query. Recall@5 and MRR are
// deliberately not saturated for the lexical offline embedder.
//
//	go test ./internal/recall -run Eval -v

// evalTargets are the hand-written items the extended queries are labelled
// against.
var evalTargets = []evalItem{
	{"t01", "Postgres serialization failure", "SQLSTATE 40001 means a serialization failure under SERIALIZABLE isolation; retry the whole transaction with backoff."},
	{"t02", "Missing grant looks like a missing table", "ORA-00942 table or view does not exist is usually a missing grant or a wrong schema prefix, not a missing table."},
	{"t03", "Uploads reset by the gateway", "ERR_CONN_RESET appears when keepalive idles longer than the load balancer timeout of sixty seconds; keep client keepalive below it."},
	{"t04", "WaitGroup Add placement", "Call wg.Add before launching the goroutine, never inside it, otherwise Wait may return before work starts."},
	{"t05", "Cancel the context to free the timer", "Always defer the cancel function returned by context.WithTimeout or the timer leaks until it fires."},
	{"t06", "Default HTTP client never times out", "http.DefaultClient waits forever on a stuck peer; construct a Client with Timeout set."},
	{"t07", "parseConfig crashes on empty YAML", "The helper parseConfig dereferences a nil map when the file is empty; guard with a length check."},
	{"t08", "Bad gateway right after a rollout", "A 502 from nginx just after deploy means upstream pods were not ready; add a readiness probe and a preStop sleep."},
	{"t09", "Move commits with rebase onto", "git rebase --onto main feature~3 feature transplants the last three commits onto main."},
	{"t10", "Recover commits with the reflog", "After a bad reset --hard, git reflog lists earlier HEAD positions so the lost commit can be checked out again."},
	{"t11", "Expired certificate", "x509: certificate has expired or is not yet valid usually means clock skew or a lapsed renewal; inspect with openssl s_client."},
	{"t12", "Shorten DNS TTL before migrating", "Lower the record TTL a day before a migration so resolvers pick up the new address quickly."},
	{"t13", "Container killed for memory", "Exit code 137 means the kernel OOM killer stopped the container for exceeding its memory limit; raise the limit or fix the leak."},
	{"t14", "Threads and the interpreter lock", "CPU bound Python threads do not run in parallel under the GIL; use multiprocessing or native extensions."},
	{"t15", "Borrow checker conflict", "E0502 cannot borrow as mutable because it is also borrowed as immutable; shorten the scope of the immutable borrow."},
	{"t16", "N plus one queries", "Loading related rows one at a time inside a loop issues N+1 queries; use a join or a batched IN query."},
	{"t17", "Idempotency keys for payments", "Send an Idempotency-Key header so a retried charge does not bill the customer twice."},
	{"t18", "Token bucket rate limiting", "A token bucket permits bursts up to its capacity while enforcing a steady average rate."},
	{"t19", "Log rotation", "Without logrotate the var log directory grows until the volume fills and services crash."},
	{"t20", "Private key permissions", "WARNING: UNPROTECTED PRIVATE KEY FILE means the key is group readable; chmod 600 the id_ed25519 file."},
	{"t21", "Docker layer caching", "Copy dependency manifests before the source so the install layer is reused across builds."},
	{"t22", "Cross origin preflight", "Browsers send an OPTIONS preflight; the server must answer with Access-Control-Allow-Origin and the allowed headers."},
	{"t23", "Store timestamps in UTC", "Persist times in UTC and convert at the edge to avoid daylight saving surprises."},
	{"t24", "Unbuffered channel deadlock", "fatal error: all goroutines are asleep - deadlock! happens when a send has no receiver."},
	{"t25", "Keyset pagination", "Offset pagination slows down on deep pages; use keyset pagination over an indexed column instead."},
	{"t26", "Retries need backoff and jitter", "Clients should retry failed calls with exponential backoff and random jitter so they do not hammer a recovering server."},
	{"t27", "Prompt injection through tools", "Treat tool output as untrusted: instructions embedded in a fetched web page must not be followed by the agent."},
	{"t28", "Verify webhook signatures", "Compare the HMAC signature header in constant time before trusting a webhook payload."},
	{"t29", "Go error wrapping", "Wrap errors with fmt.Errorf and the %w verb so callers can use errors.Is and errors.As."},
	{"t30", "Close response bodies", "Always close resp.Body after http.Get, otherwise connections are never returned to the pool."},
}

// evalNearMisses share identifiers or phrasing with the targets without
// being relevant to any query.
var evalNearMisses = []evalItem{
	{"x01", "Deadlock detected", "SQLSTATE 40P01 means the database detected a deadlock and aborted one transaction."},
	{"x02", "Invalid identifier", "ORA-00904 invalid identifier is raised when a column name is misspelled or quoted wrongly."},
	{"x03", "Double mutable borrow", "E0499 cannot borrow as mutable more than once at a time; split the struct into disjoint fields."},
	{"x04", "Graceful shutdown", "Exit code 143 means the process received SIGTERM; handle it and drain connections before exiting."},
	{"x05", "Gateway timeout", "A 504 from the proxy means the upstream was too slow, not that it was unreachable."},
	{"x06", "Connection refused", "ECONNREFUSED means nothing is listening on the port; check the service and the firewall."},
	{"x07", "Merge strategies", "git merge --no-ff keeps a merge commit; rebase rewrites history instead of recording a merge."},
	{"x08", "Stash changes", "git stash saves a dirty working tree so you can switch branches and come back later."},
	{"x09", "Client timeouts for gRPC", "Set a deadline on every gRPC call so a stuck server cannot hold the caller forever."},
	{"x10", "Goroutine leaks", "A goroutine blocked on a channel nobody reads leaks for the life of the process."},
	{"x11", "Config loader tests", "The config loader tests cover empty files, missing keys and invalid YAML syntax."},
	{"x12", "Certificate rotation runbook", "Renew the TLS certificate thirty days before expiry and reload the proxy afterwards."},
	{"x13", "Hash passwords with salt", "Use a per-user salt and a slow hash so rainbow tables are useless."},
	{"x14", "Billing retries", "Failed card payments are retried three times over a week before the subscription lapses."},
	{"x15", "Cache key naming", "Prefix cache keys with the service name and version so deploys never read stale shapes."},
}

// evalExtraQueries are labelled against evalTargets (and the base corpus).
var evalExtraQueries = []evalQuery{
	// Exact identifiers.
	{"SQLSTATE 40001", []string{"t01"}},
	{"ORA-00942", []string{"t02"}},
	{"E0502", []string{"t15"}},
	{"ERR_CONN_RESET on uploads", []string{"t03"}},
	{"parseConfig nil map panic", []string{"t07"}},
	{"context.WithTimeout cancel", []string{"t05"}},
	{"git rebase --onto", []string{"t09"}},
	{"x509 certificate has expired", []string{"t11"}},
	{"all goroutines are asleep", []string{"t24"}},
	{"what does exit code 137 mean", []string{"t13"}},
	// Natural language full of stopwords.
	{"what should I do when the transaction fails with a serialization error", []string{"t01"}},
	{"how do I make sure that the goroutine has actually started before we wait", []string{"t04"}},
	{"why does the request to the other server hang for ever", []string{"t06"}},
	{"what is the best way to get back the commits that I lost", []string{"t10"}},
	{"how can I stop the pods from returning errors when we ship a new version", []string{"t08"}},
	{"is there a way to avoid the same payment being taken twice", []string{"t17"}},
	{"how should the client behave when the server is overloaded and calls fail", []string{"t26"}},
	{"how do I know that a message really was sent by the provider", []string{"t28"}},
	// Paraphrases with little lexical overlap.
	{"my threads are not any faster in python", []string{"t14"}},
	{"orm fires hundreds of statements for a single page", []string{"t16"}},
	{"allow short spikes but cap the long run rate", []string{"t18"}},
	{"disk is full because of old log files", []string{"t19"}},
	{"ssh refuses my key because the file is too open", []string{"t20"}},
	{"rebuilds are slow when only the source changed", []string{"t21"}},
	{"browser blocks my call to the API on another domain", []string{"t22"}},
	{"timestamps are an hour off twice a year", []string{"t23"}},
	{"page 5000 of the results takes forever", []string{"t25"}},
	{"a fetched web page tells the agent to ignore its rules", []string{"t27"}},
	{"how do I let callers check for a specific wrapped error", []string{"t29"}},
	{"connections run out because I never finish reading", []string{"t30"}},
	{"make the switch to a new IP address take effect quickly", []string{"t12"}},
}

// fillerWords are the pools the seeded filler generator draws from. They
// are generic engineering prose full of stopwords, deliberately sharing
// words ("database", "slow", "fix", "retry", "timeout", ...) with queries.
var (
	fillerSubjects = []string{"the database", "the cache layer", "the API gateway", "the worker pool", "the build pipeline",
		"the scheduler", "the message queue", "the auth service", "the search index", "the deploy job",
		"the load balancer", "the config loader", "the metrics exporter", "the session store", "the file uploader",
		"the billing job", "the web client", "the mobile app", "the batch importer", "the log shipper"}
	fillerProblems = []string{"is slow under load", "keeps timing out", "returns stale results", "leaks memory over time",
		"fails after a restart", "drops requests during a rollout", "uses too much disk", "retries too aggressively",
		"logs too little to debug", "blocks on a lock", "needs a fix before the release", "was changed by the last migration",
		"does not retry the failed request", "waits for the slow dependency", "cannot reach the database"}
	fillerActions = []string{"check the metrics dashboard first", "raise the limit gradually and measure", "add a regression test before changing it",
		"compare with the last known good release", "read the logs around the first failure", "roll back and bisect the change",
		"ask the owning team before touching it", "document the fix in the runbook", "look at the slow query log",
		"verify the config in the staging environment", "restart one instance and watch the error rate"}
	fillerTemplates = []string{
		"When %s %s, %s.",
		"We noticed that %s %s. The usual remedy is to %s.",
		"%s sometimes %s; %s and then verify the result.",
		"Notes from the last incident: %s %s, so %s.",
	}
)

// genFiller returns n deterministic filler items (ids z0000...).
func genFiller(n int, seed int64) []evalItem {
	rng := rand.New(rand.NewSource(seed))
	out := make([]evalItem, 0, n)
	for i := 0; i < n; i++ {
		subj := fillerSubjects[rng.Intn(len(fillerSubjects))]
		prob := fillerProblems[rng.Intn(len(fillerProblems))]
		act := fillerActions[rng.Intn(len(fillerActions))]
		tpl := fillerTemplates[rng.Intn(len(fillerTemplates))]
		body := fmt.Sprintf(tpl, subj, prob, act)
		if rng.Intn(4) == 0 { // an unrelated error code, to crowd identifier queries
			body += fmt.Sprintf(" It showed error E%04d in the %s.", 1000+rng.Intn(9000), subj[4:])
		}
		title := "Note " + subj[4:] + " " + prob
		out = append(out, evalItem{fmt.Sprintf("z%04d", i), title, body})
	}
	return out
}

// extendedEvalCorpus is the base corpus, targets, near misses and filler.
func extendedEvalCorpus() []evalItem {
	var all []evalItem
	all = append(all, evalCorpus...)
	all = append(all, evalTargets...)
	all = append(all, evalNearMisses...)
	all = append(all, genFiller(300, 42)...)
	return all
}

func extendedEvalQueries() []evalQuery {
	return append(append([]evalQuery{}, evalQueries...), evalExtraQueries...)
}

// scoreEval runs the queries and returns recall@5 and MRR.
func scoreEval(t *testing.T, eng *Engine, queries []evalQuery) (recall5, mrr float64) {
	t.Helper()
	for _, q := range queries {
		res, err := eng.Run(context.Background(), Query{Goal: q.goal, TopK: 10})
		if err != nil {
			t.Fatalf("%q: %v", q.goal, err)
		}
		rel := map[string]bool{}
		for _, id := range q.relevant {
			rel[id] = true
		}
		found, first := 0, 0
		for rank, r := range res {
			if !rel[r.ID] {
				continue
			}
			if rank < 5 {
				found++
			}
			if first == 0 {
				first = rank + 1
			}
		}
		recall5 += float64(found) / float64(len(q.relevant))
		if first > 0 {
			mrr += 1 / float64(first)
		}
	}
	n := float64(len(queries))
	return recall5 / n, mrr / n
}

// TestEval_FusionModesExtended scores every fusion mode on the extended
// corpus. Only loose lower bounds are asserted.
func TestEval_FusionModesExtended(t *testing.T) {
	if f := flag.Lookup("test.run"); f == nil || !strings.Contains(f.Value.String(), "Eval") {
		t.Skip("evaluation harness: run with -run Eval")
	}
	corpus, queries := extendedEvalCorpus(), extendedEvalQueries()
	for _, mode := range []string{"legacy", "minmax", "rrf"} {
		env := newTestEnv(t, embed.NewOfflineEmbedder(256), mode)
		for i, it := range corpus {
			env.insertItem(t, it.id, "fact", it.title, it.body, nil, 0.6+0.05*float64(i%8))
		}
		r5, mrr := scoreEval(t, env.eng, queries)
		t.Logf("%-7s items %d queries %d  recall@5 %.3f  MRR %.3f", mode, len(corpus), len(queries), r5, mrr)
		if r5 < 0.4 || mrr < 0.4 {
			t.Errorf("%s: recall@5 %.3f / MRR %.3f below the loose floors (0.4)", mode, r5, mrr)
		}
	}
}
