package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	// cooldownPeriod is how long an embedder is skipped after a
	// connection-level/transient failure (retries exhausted) or a 501.
	// Callers get a fast errCoolingDown instead of paying the retry
	// backoff again, so a missing llama-server costs one probe per window
	// rather than seconds per Recall/Remember.
	cooldownPeriod = 30 * time.Second

	// connRefusedWait is the single short wait before the one allowed
	// retry of a refused connection. A refused connection on localhost is
	// immediate and means nothing is listening, so long backoff is pointless.
	connRefusedWait = 100 * time.Millisecond
)

// errCoolingDown marks a fast failure returned while an embedder is in its
// cool-down window (or another caller is already probing it).
var errCoolingDown = errors.New("embedder cooling down after earlier failure")

// statusError is a non-2xx HTTP answer. Its text is "status <code>: <body>".
type statusError struct {
	code   int
	detail string
}

func (e *statusError) Error() string { return fmt.Sprintf("status %d: %s", e.code, e.detail) }

// tripError marks a failure that should start the cool-down: the server
// is unreachable or answering with a 5xx (including a permanent 501).
type tripError struct{ err error }

func (e *tripError) Error() string { return e.err.Error() }
func (e *tripError) Unwrap() error { return e.err }

// retryableStatus reports whether an HTTP status is worth retrying: 429
// and 5xx, except the permanent answers 501 (llama-server without
// --embeddings) and 505 (HTTP version not supported).
func retryableStatus(code int) bool {
	if code == http.StatusNotImplemented || code == http.StatusHTTPVersionNotSupported {
		return false
	}
	return code == http.StatusTooManyRequests || code >= 500
}

// retryAfter parses a Retry-After header given in seconds (the HTTP-date
// form is ignored).
func retryAfter(h http.Header) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(h.Get("Retry-After")))
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs) * time.Second
}

// retryWait is the delay before retry number attempt (>= 1): quadratic
// backoff, or the server's Retry-After when that is longer.
func retryWait(attempt int, after time.Duration) time.Duration {
	return max(time.Duration(attempt*attempt)*250*time.Millisecond, after)
}

// retryDo calls do until it succeeds, reports a non-retryable error, the
// attempt budget (maxRetries+1) is spent, or ctx ends. do reports whether
// its error is retryable and any server-requested Retry-After.
//
// Connection-refused is retried at most once, after connRefusedWait.
// The returned error is wrapped in a *tripError when it should start the
// cool-down; caller cancellation never does.
func retryDo(ctx context.Context, maxRetries int, do func() ([][]float32, bool, time.Duration, error)) ([][]float32, error) {
	attempts := max(maxRetries+1, 1)
	var (
		lastErr   error
		after     time.Duration
		retryable bool
		refused   bool
	)
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			wait := retryWait(attempt, after)
			if refused {
				wait = connRefusedWait
			}
			t := time.NewTimer(wait)
			select {
			case <-ctx.Done():
				t.Stop()
				return nil, ctx.Err()
			case <-t.C:
			}
		}
		var vecs [][]float32
		vecs, retryable, after, lastErr = do()
		if lastErr == nil {
			return vecs, nil
		}
		if ctx.Err() != nil {
			return nil, lastErr // the caller gave up; not the server's fault
		}
		refused = errors.Is(lastErr, syscall.ECONNREFUSED)
		if !retryable || (refused && attempt >= 1) {
			break
		}
	}
	var se *statusError
	if retryable || (errors.As(lastErr, &se) && se.code >= 500) {
		return nil, &tripError{lastErr}
	}
	return nil, lastErr
}

// cooldown skips a failing embedder for cooldownPeriod. The zero value is
// ready to use. While tripped, calls fail fast; once the window elapses a
// single probe call is let through and concurrent callers keep failing fast
// until it finishes. Success resets the state.
type cooldown struct {
	now func() time.Time // test hook; nil means time.Now

	mu      sync.Mutex
	until   time.Time // zero: healthy
	probing bool
	last    error
}

func (c *cooldown) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// run executes fn unless the embedder is cooling down.
func (c *cooldown) run(fn func() ([][]float32, error)) ([][]float32, error) {
	probe, err := c.acquire()
	if err != nil {
		return nil, err
	}
	vecs, err := fn()
	c.release(probe, err)
	return vecs, err
}

func (c *cooldown) acquire() (probe bool, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.until.IsZero() {
		return false, nil
	}
	if left := c.until.Sub(c.clock()); left > 0 || c.probing {
		left = max(left, 0)
		return false, fmt.Errorf("gomem: %w (%s left; last error: %v)", errCoolingDown, left.Round(time.Second), c.last)
	}
	c.probing = true
	return true, nil
}

func (c *cooldown) release(probe bool, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if probe {
		c.probing = false
	}
	var te *tripError
	switch {
	case err == nil:
		c.until, c.last = time.Time{}, nil
	case errors.As(err, &te):
		c.until, c.last = c.clock().Add(cooldownPeriod), te.err
	}
}

// remote is the transport shared by the HTTP embedders: batching, retries,
// the cool-down and the POST itself. An embedder supplies only its request
// and response shapes.
type remote struct {
	url, apiKey string
	maxRetries  int
	batchSize   int
	errPrefix   string // prepended to a failed embed's error
	hc          *http.Client
	cd          cooldown // skips the endpoint briefly after it fails
}

// positive returns v, or def when v is not positive.
func positive[T ~int | ~int64](v, def T) T {
	if v > 0 {
		return v
	}
	return def
}

func newRemote(url, apiKey string, timeout time.Duration, maxRetries, batchSize int, errPrefix string) remote {
	return remote{
		url: url, apiKey: apiKey, maxRetries: maxRetries, errPrefix: errPrefix,
		batchSize: positive(batchSize, 32), hc: &http.Client{Timeout: timeout},
	}
}

// embed sends texts in batches, retrying transient failures and failing
// fast while the endpoint cools down after a connection-level failure or
// 5xx. request builds a batch's JSON body; parse decodes a response into
// the vectors for a batch of n texts (a nil entry marks a missing one).
func (r *remote) embed(ctx context.Context, texts []string, request func(batch []string) any,
	parse func(raw []byte, n int) ([][]float32, error)) ([][]float32, error) {
	return r.cd.run(func() ([][]float32, error) {
		out := make([][]float32, len(texts))
		for start := 0; start < len(texts); start += r.batchSize {
			batch := texts[start:min(start+r.batchSize, len(texts))]
			body, err := json.Marshal(request(batch))
			if err != nil {
				return nil, err
			}
			vecs, err := retryDo(ctx, r.maxRetries, func() ([][]float32, bool, time.Duration, error) {
				return r.post(ctx, body, len(batch), parse)
			})
			if err != nil {
				return nil, fmt.Errorf("%s: %w", r.errPrefix, err)
			}
			copy(out[start:], vecs)
		}
		return out, nil
	})
}

// post performs one request. Besides the vectors it reports whether a
// failure is retryable and any server-requested Retry-After delay.
func (r *remote) post(ctx context.Context, body []byte, n int,
	parse func([]byte, int) ([][]float32, error)) ([][]float32, bool, time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.url, bytes.NewReader(body))
	if err != nil {
		return nil, false, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.apiKey)
	}
	resp, err := r.hc.Do(req)
	if err != nil {
		return nil, true, 0, err // network error: retryable
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, retryableStatus(resp.StatusCode), retryAfter(resp.Header), &statusError{resp.StatusCode, string(detail)}
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, false, 0, err
	}
	vecs, err := parse(raw, n)
	return vecs, false, 0, err
}
