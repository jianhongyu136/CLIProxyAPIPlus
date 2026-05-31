package toolemu

import (
	"strings"
	"testing"
)

func TestDeriveID_Deterministic(t *testing.T) {
	m := UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r"}
	a := DeriveID(m, 0)
	b := DeriveID(m, 0)
	if a != b {
		t.Fatal("same inputs must yield same id")
	}
}

func TestDeriveID_IndexDifferentiates(t *testing.T) {
	m := UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r"}
	if DeriveID(m, 0) == DeriveID(m, 1) {
		t.Fatal("different indexes must yield different ids")
	}
}

func TestDeriveID_FormatPrefix(t *testing.T) {
	id := DeriveID(UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r"}, 0)
	if !strings.HasPrefix(id, "call_") {
		t.Fatalf("missing prefix: %q", id)
	}
	if len(id) != len("call_")+8 {
		t.Fatalf("unexpected length: %q", id)
	}
}

func TestDeriveID_EmptyResponseIDFallback(t *testing.T) {
	id := DeriveID(UpstreamMeta{Provider: "p", Model: "m"}, 0)
	if !strings.HasPrefix(id, "call_") || len(id) != len("call_")+8 {
		t.Fatalf("fallback id malformed: %q", id)
	}
}
