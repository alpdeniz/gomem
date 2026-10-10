package gomem

import (
	"context"
	"fmt"
	"testing"
)

func TestIsTransient_Nil(t *testing.T) {
	if IsTransient(nil) {
		t.Fatal("nil error must not be transient")
	}
}

func TestIsTransient_Canceled(t *testing.T) {
	if IsTransient(context.Canceled) {
		t.Fatal("context.Canceled must not be transient")
	}
	if IsTransient(fmt.Errorf("embed: %w", context.Canceled)) {
		t.Fatal("wrapped context.Canceled must not be transient")
	}
	if !IsTransient(context.DeadlineExceeded) {
		t.Fatal("context.DeadlineExceeded should stay transient")
	}
}
