package embed

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeClock is an injectable cooldown clock so tests never sleep.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock() *fakeClock { return &fakeClock{t: time.Unix(1_700_000_000, 0)} }

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	c.t = c.t.Add(d)
	c.mu.Unlock()
}

// statusServer answers every request with the current status (200 emits a
// llama-server style single-vector body) and counts requests.
type statusServer struct {
	*httptest.Server
	status atomic.Int32
	reqs   atomic.Int32
}

func newStatusServer(t *testing.T, status int) *statusServer {
	t.Helper()
	s := &statusServer{}
	s.status.Store(int32(status))
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.reqs.Add(1)
		if code := int(s.status.Load()); code != http.StatusOK {
			w.WriteHeader(code)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embedding":[1,2],"data":[{"index":0,"embedding":[1,2]}]}`))
	}))
	t.Cleanup(s.Close)
	return s
}

// deadURL returns a URL whose port has nothing listening (refused).
func deadURL(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	url := "http://" + ln.Addr().String()
	_ = ln.Close()
	return url
}

// embedderUnderTest lets the same scenarios run against both embedders.
type embedderUnderTest struct {
	name string
	make func(url string, retries int, clk *fakeClock) Embedder
}

var embedders = []embedderUnderTest{
	{"llamacpp", func(url string, retries int, clk *fakeClock) Embedder {
		e := NewLlamaCppEmbedder(LlamaCppConfig{ServerURL: url, Dim: 2, MaxRetries: retries})
		e.cd.now = clk.Now
		return e
	}},
	{"api", func(url string, retries int, clk *fakeClock) Embedder {
		e := NewAPIEmbedder(APIConfig{APIURL: url, Dim: 2, MaxRetries: retries})
		e.cd.now = clk.Now
		return e
	}},
}

func TestRetryableStatus(t *testing.T) {
	for code, want := range map[int]bool{
		200: false, 400: false, 404: false, 429: true,
		500: true, 501: false, 502: true, 503: true, 504: true, 505: false,
	} {
		if got := retryableStatus(code); got != want {
			t.Errorf("retryableStatus(%d) = %v, want %v", code, got, want)
		}
	}
}

// A refused connection must cost at most one short retry (not the full
// 0.25+1+2.25 s backoff), and the next call must fast-fail with no network.
func TestEmbedder_RefusedConnectionFailsFastThenCoolsDown(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			url := deadURL(t)
			if tc.name == "llamacpp" {
				url += "/"
			}
			clk := newFakeClock()
			emb := tc.make(url, 3, clk) // default-like retry budget

			start := time.Now()
			if _, err := emb.Embed(context.Background(), []string{"x"}); err == nil || errors.Is(err, errCoolingDown) {
				t.Fatalf("first call: want a connection error, got %v", err)
			}
			if d := time.Since(start); d > time.Second {
				t.Fatalf("first failure took %v, want < 1s", d)
			}

			start = time.Now()
			_, err := emb.Embed(context.Background(), []string{"x"})
			if !errors.Is(err, errCoolingDown) {
				t.Fatalf("second call: want errCoolingDown, got %v", err)
			}
			if d := time.Since(start); d > 50*time.Millisecond {
				t.Fatalf("cool-down call took %v, want immediate", d)
			}
		})
	}
}

// 5xx: tripped after retries are exhausted, fast-fails without touching the
// network, probes once after the window, and success resets the state.
func TestEmbedder_CooldownProbeAndReset(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			srv := newStatusServer(t, http.StatusServiceUnavailable)
			clk := newFakeClock()
			emb := tc.make(srv.URL, 0, clk)
			ctx := context.Background()

			if _, err := emb.Embed(ctx, []string{"x"}); err == nil || errors.Is(err, errCoolingDown) {
				t.Fatalf("want a 503 failure, got %v", err)
			}
			for i := 0; i < 5; i++ {
				if _, err := emb.Embed(ctx, []string{"x"}); !errors.Is(err, errCoolingDown) {
					t.Fatalf("call %d in window: want errCoolingDown, got %v", i, err)
				}
			}
			if n := srv.reqs.Load(); n != 1 {
				t.Fatalf("requests during cool-down = %d, want 1", n)
			}

			clk.Advance(cooldownPeriod - time.Second)
			if _, err := emb.Embed(ctx, []string{"x"}); !errors.Is(err, errCoolingDown) {
				t.Fatalf("just before the window ends: want errCoolingDown, got %v", err)
			}

			// Window over: one probe, still failing -> window restarts.
			clk.Advance(time.Second)
			if _, err := emb.Embed(ctx, []string{"x"}); err == nil || errors.Is(err, errCoolingDown) {
				t.Fatalf("probe: want a 503 failure, got %v", err)
			}
			if n := srv.reqs.Load(); n != 2 {
				t.Fatalf("requests after probe = %d, want 2", n)
			}
			if _, err := emb.Embed(ctx, []string{"x"}); !errors.Is(err, errCoolingDown) {
				t.Fatalf("after failed probe: want errCoolingDown, got %v", err)
			}

			// Server recovers; next probe succeeds and resets everything.
			srv.status.Store(http.StatusOK)
			clk.Advance(cooldownPeriod)
			if _, err := emb.Embed(ctx, []string{"x"}); err != nil {
				t.Fatalf("recovery probe: %v", err)
			}
			for i := 0; i < 3; i++ {
				if _, err := emb.Embed(ctx, []string{"x"}); err != nil {
					t.Fatalf("after reset call %d: %v", i, err)
				}
			}
			if n := srv.reqs.Load(); n != 6 {
				t.Fatalf("total requests = %d, want 6", n)
			}
		})
	}
}

// 501 is a permanent "embeddings not enabled": exactly one request despite
// the retry budget, and it trips the cool-down immediately.
func TestEmbedder_501NotRetriedAndTripsCooldown(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			srv := newStatusServer(t, http.StatusNotImplemented)
			emb := tc.make(srv.URL, 3, newFakeClock())
			if _, err := emb.Embed(context.Background(), []string{"x"}); err == nil {
				t.Fatal("want an error for 501")
			}
			if n := srv.reqs.Load(); n != 1 {
				t.Fatalf("requests = %d, want exactly 1", n)
			}
			if _, err := emb.Embed(context.Background(), []string{"x"}); !errors.Is(err, errCoolingDown) {
				t.Fatalf("want errCoolingDown after 501, got %v", err)
			}
			if n := srv.reqs.Load(); n != 1 {
				t.Fatalf("requests after cool-down call = %d, want 1", n)
			}
		})
	}
}

// 4xx client errors never start a cool-down.
func TestEmbedder_4xxDoesNotCoolDown(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			srv := newStatusServer(t, http.StatusBadRequest)
			emb := tc.make(srv.URL, 3, newFakeClock())
			for i := 0; i < 3; i++ {
				if _, err := emb.Embed(context.Background(), []string{"x"}); err == nil || errors.Is(err, errCoolingDown) {
					t.Fatalf("call %d: want a 400 failure, got %v", i, err)
				}
			}
			if n := srv.reqs.Load(); n != 3 {
				t.Fatalf("requests = %d, want 3 (one per call)", n)
			}
		})
	}
}

// Caller cancellation is not the server's fault and must not trip the
// cool-down.
func TestEmbedder_ContextCancelDoesNotCoolDown(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			var reqs atomic.Int32
			entered := make(chan struct{}, 1)
			stop := make(chan struct{}) // closed before srv.Close so a missed disconnect can't hang the test
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body) // lets the server notice a client disconnect
				if reqs.Add(1) == 1 {
					entered <- struct{}{}
					select { // hang until the client gives up
					case <-r.Context().Done():
					case <-stop:
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"embedding":[1,2],"data":[{"index":0,"embedding":[1,2]}]}`))
			}))
			defer srv.Close()
			defer close(stop)
			emb := tc.make(srv.URL, 3, newFakeClock())

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			go func() { <-entered; cancel() }()
			if _, err := emb.Embed(ctx, []string{"x"}); err == nil {
				t.Fatal("want an error after cancellation")
			}
			if _, err := emb.Embed(context.Background(), []string{"x"}); err != nil {
				t.Fatalf("call after cancel must go through, got %v", err)
			}
		})
	}
}

// After the window a single probe is admitted; concurrent callers fast-fail
// while it is in flight. Also exercised under -race.
func TestEmbedder_ConcurrentCallersDuringProbeFastFail(t *testing.T) {
	for _, tc := range embedders {
		t.Run(tc.name, func(t *testing.T) {
			var reqs atomic.Int32
			entered := make(chan struct{})
			release := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch reqs.Add(1) {
				case 1: // initial failure trips the cool-down
					w.WriteHeader(http.StatusBadGateway)
					return
				case 2: // the probe: hold it open
					close(entered)
					<-release
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"embedding":[1,2],"data":[{"index":0,"embedding":[1,2]}]}`))
			}))
			defer srv.Close()
			clk := newFakeClock()
			emb := tc.make(srv.URL, 0, clk)

			if _, err := emb.Embed(context.Background(), []string{"x"}); err == nil {
				t.Fatal("want 502 failure")
			}
			clk.Advance(cooldownPeriod)

			probeErr := make(chan error, 1)
			go func() {
				_, err := emb.Embed(context.Background(), []string{"x"})
				probeErr <- err
			}()
			<-entered // the probe is now blocked inside the server

			var wg sync.WaitGroup
			for i := 0; i < 16; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					if _, err := emb.Embed(context.Background(), []string{"x"}); !errors.Is(err, errCoolingDown) {
						t.Errorf("concurrent caller: want errCoolingDown, got %v", err)
					}
				}()
			}
			wg.Wait()
			close(release)
			if err := <-probeErr; err != nil {
				t.Fatalf("probe: %v", err)
			}
			if n := reqs.Load(); n != 2 {
				t.Fatalf("requests = %d, want 2 (initial + single probe)", n)
			}
			if _, err := emb.Embed(context.Background(), []string{"x"}); err != nil {
				t.Fatalf("after successful probe: %v", err)
			}
		})
	}
}
