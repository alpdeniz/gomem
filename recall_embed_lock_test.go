package gomem_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/alpdeniz/gomem"
	"github.com/alpdeniz/gomem/internal/testutil"
)

// TestRecall_SlowEmbedDoesNotHoldLock: Recall embeds its goal before taking
// the database lock, so a slow embed stalls neither writers nor (behind a
// pending writer) readers.
func TestRecall_SlowEmbedDoesNotHoldLock(t *testing.T) {
	ctx := context.Background()
	emb := newGateEmbedder()
	m, done := newTestMemory(t, emb)
	defer done()
	var once sync.Once
	releaseEmbed := func() { once.Do(func() { close(emb.release) }) }
	defer releaseEmbed() // on failure, unblock a stuck Recall so done() can Close
	seedItems(t, ctx, m)

	recallDone := make(chan error, 1)
	go func() {
		_, err := m.Recall(ctx, gomem.RecallQuery{Goal: "BLOCK composite index database", TopK: 3})
		recallDone <- err
	}()
	<-emb.entered // the Recall is now inside the embedder

	finished := make(chan error, 2)
	go func() {
		_, err := m.Remember(ctx, gomem.Item{Kind: gomem.KindFact, Title: "written during embed", Body: "body"})
		finished <- err
	}()
	go func() {
		_, err := m.Stats(ctx)
		finished <- err
	}()
	for i := 0; i < 2; i++ {
		select {
		case err := <-finished:
			testutil.AssertNoError(t, err)
		case <-time.After(5 * time.Second):
			t.Fatal("a Remember/Stats was blocked behind a Recall that is still embedding")
		}
	}
	select {
	case <-recallDone:
		t.Fatal("Recall returned before the embedder was released")
	default:
	}
	releaseEmbed()
	testutil.AssertNoError(t, <-recallDone)
}
