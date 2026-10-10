package recall

import (
	"context"
	"flag"
	"strings"
	"testing"

	"github.com/alpdeniz/gomem/internal/embed"
)

// The fusion evaluation harness scores every fusion mode on a small
// labelled corpus (recall@5 and MRR). It only runs when selected, so the
// regular suite stays fast:
//
//	go test ./internal/recall -run Eval -v

type evalItem struct{ id, title, body string }

type evalQuery struct {
	goal     string
	relevant []string
}

var evalCorpus = []evalItem{
	{"a1", "Composite indexes accelerate filtered queries", "Align a composite index with the WHERE and ORDER BY columns so the database avoids full table scans and sorts."},
	{"a2", "Covering index avoids table lookups", "When an index contains every column the query reads, the database answers from the index alone without touching table rows."},
	{"a3", "Use EXPLAIN to inspect query plans", "Run EXPLAIN before tuning: it shows whether the planner uses an index or falls back to a sequential scan."},
	{"a4", "Batch writes inside one transaction", "Wrapping many inserts in a single transaction cuts fsync overhead and speeds up bulk loads dramatically."},
	{"a5", "Connection pooling limits database load", "A bounded pool of reusable connections prevents thundering herds of short lived clients from exhausting the server."},
	{"b1", "Cache hot reads to cut latency", "Memoizing frequently requested, slowly changing results lowers response time and CPU spend."},
	{"b2", "Invalidate cache entries on write", "Stale data appears when writers forget to evict or update the cached copy; tie invalidation to the write path."},
	{"b3", "Set a time-to-live on cached values", "A TTL bounds staleness: entries expire automatically so the cache heals even if invalidation is missed."},
	{"b4", "Use a CDN for static assets", "Edge servers close to users serve images and scripts quickly and take load off the origin."},
	{"b5", "Cache stampede protection", "When a popular key expires, many requests recompute it at once; use request coalescing or jittered expiry."},
	{"c1", "Rotate bearer tokens regularly", "Short-lived access tokens with refresh rotation limit the damage when a token leaks."},
	{"c2", "Hash passwords with a slow algorithm", "Use bcrypt or argon2 with a per-user salt so stolen hashes are expensive to crack."},
	{"c3", "Enable multi-factor authentication", "A second factor such as a hardware key blocks account takeover even after password theft."},
	{"c4", "Validate input to prevent injection", "Parameterized queries and strict validation stop SQL injection and script injection attacks."},
	{"c5", "Principle of least privilege", "Grant each service account only the permissions it needs so a compromise stays contained."},
	{"d1", "Feed the sourdough starter on a schedule", "Refresh your starter with equal parts flour and water every twelve hours to keep the yeast active."},
	{"d2", "Cold ferment dough overnight", "A slow refrigerated rise develops flavor and a more open crumb in artisan bread."},
	{"d3", "Score the loaf before baking", "A sharp blade slash lets steam escape and controls where the crust opens in the oven."},
	{"d4", "Bake in a preheated Dutch oven", "Trapped steam from a covered cast iron pot yields a crackling crust and good oven spring."},
	{"d5", "Hydration affects crumb texture", "Wetter dough produces larger holes but is harder to shape; start around seventy percent water."},
	{"e1", "Pack layers for changing mountain weather", "Bring a base layer, insulating fleece and a waterproof shell because temperatures swing fast at altitude."},
	{"e2", "Break in boots before a long hike", "Wear new hiking boots on short walks first to prevent blisters on the trail."},
	{"e3", "Carry a map and compass", "Batteries die and signal drops; paper navigation tools keep you oriented in the backcountry."},
	{"e4", "Store food away from your tent", "Hang food or use a bear canister to keep animals away from your campsite."},
	{"e5", "Filter or treat stream water", "Use a filter or purification tablets to remove parasites before drinking water from rivers."},
	{"f1", "Set resource requests and limits", "Kubernetes pods without CPU and memory limits can starve neighbours and trigger evictions."},
	{"f2", "Use readiness probes before routing traffic", "A readiness probe keeps a pod out of the service until the application can actually answer requests."},
	{"f3", "Roll out with a canary release", "Send a small share of traffic to the new version and watch error rates before promoting it fully."},
	{"f4", "Store secrets outside container images", "Mount credentials from a secret store at runtime instead of baking them into the image."},
	{"f5", "Autoscale pods on CPU load", "A horizontal pod autoscaler adds replicas when average CPU utilization crosses the target."},
	{"g1", "Build an emergency fund first", "Keep three to six months of expenses in a savings account before investing."},
	{"g2", "Track spending with a monthly budget", "Categorize expenses each month to find leaks and direct money toward goals."},
	{"g3", "Pay off high interest debt early", "Credit card interest compounds faster than most investments earn, so clear it before investing."},
	{"g4", "Diversify investments across assets", "Spreading money over stocks, bonds and funds reduces the impact of any single loss."},
	{"g5", "Contribute to retirement accounts early", "Compound growth rewards early contributions, especially when an employer matches deposits."},
	{"h1", "Hold out a validation set", "Evaluate on data the model never trained on to detect overfitting."},
	{"h2", "Tune the learning rate first", "A learning rate that is too high diverges while too low trains slowly; sweep it before other hyperparameters."},
	{"h3", "Normalize input features", "Scaling features to zero mean and unit variance helps gradient descent converge faster."},
	{"h4", "Use early stopping", "Stop training when validation loss stops improving to avoid memorizing the training set."},
	{"h5", "Augment data to reduce overfitting", "Random flips and crops enlarge the effective training set for image models."},
	{"i1", "Increase weekly mileage gradually", "Raise running volume by no more than ten percent a week to avoid stress injuries."},
	{"i2", "Warm up before intervals", "Easy jogging and dynamic drills prepare muscles for fast repeats and lower strain risk."},
	{"i3", "Replace running shoes regularly", "Cushioning degrades after several hundred miles, so swap shoes to protect your joints."},
	{"i4", "Fuel long runs with carbohydrates", "Take gels or chews every forty minutes during runs longer than ninety minutes."},
	{"i5", "Rest days support recovery", "Muscles rebuild during rest; schedule easy days to improve performance."},
	// Distractors that share words with the topics above.
	{"j1", "Back-of-book index entries", "An index at the end of a book lists topics with page numbers so readers can find passages quickly."},
	{"j2", "Geocaching hidden cache", "A geocache is a hidden container of small trinkets that hikers locate with a GPS device."},
	{"j3", "Tokenization splits text into tokens", "Language models break sentences into subword tokens before computing embeddings."},
	{"j4", "Wine cellar storage conditions", "Keep bottles at steady cool temperature and humidity so the cork stays sealed."},
	{"j5", "Library catalog search", "A librarian queries the catalog by author title or subject to locate shelf numbers."},
}

var evalQueries = []evalQuery{
	{"speed up slow database queries with an index", []string{"a1", "a2"}},
	{"how do I see whether my query uses an index or scans the table", []string{"a3"}},
	{"bulk insert performance", []string{"a4"}},
	{"too many clients exhausting the database server", []string{"a5"}},
	{"cached data is stale after updates", []string{"b2", "b3"}},
	{"reduce latency by caching frequently read results", []string{"b1"}},
	{"popular key expired and everything recomputes", []string{"b5"}},
	{"protect accounts against stolen passwords", []string{"c2", "c3"}},
	{"what to do when an access token leaks", []string{"c1"}},
	{"stop SQL injection", []string{"c4"}},
	{"keep starter yeast alive", []string{"d1"}},
	{"get a crispy crust on homemade bread", []string{"d3", "d4"}},
	{"make bread taste better with a long rise", []string{"d2"}},
	{"avoid blisters on a long walk in the mountains", []string{"e2"}},
	{"stay safe from bears at camp", []string{"e4"}},
	{"drink safely from a creek", []string{"e5"}},
	{"pods getting evicted for lack of memory", []string{"f1"}},
	{"gradually deploy a new version and watch errors", []string{"f3"}},
	{"my model memorizes the training data", []string{"h1", "h4", "h5"}},
	{"how fast should I increase running distance", []string{"i1"}},
}

// TestEval_FusionModes reports recall@5 and MRR per fusion mode.
func TestEval_FusionModes(t *testing.T) {
	if f := flag.Lookup("test.run"); f == nil || !strings.Contains(f.Value.String(), "Eval") {
		t.Skip("evaluation harness: run with -run Eval")
	}
	ctx := context.Background()
	for _, mode := range []string{"legacy", "minmax", "rrf"} {
		env := newTestEnv(t, embed.NewOfflineEmbedder(256), mode)
		for i, it := range evalCorpus {
			env.insertItem(t, it.id, "fact", it.title, it.body, nil, 0.6+0.05*float64(i%8))
		}
		var recall5, mrr float64
		for _, q := range evalQueries {
			res, err := env.eng.Run(ctx, Query{Goal: q.goal, TopK: 10})
			if err != nil {
				t.Fatalf("%s %q: %v", mode, q.goal, err)
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
		n := float64(len(evalQueries))
		t.Logf("%-7s recall@5 %.3f  MRR %.3f", mode, recall5/n, mrr/n)
		if recall5/n < 0.8 {
			t.Errorf("%s: recall@5 %.3f below the 0.8 floor", mode, recall5/n)
		}
	}
}
