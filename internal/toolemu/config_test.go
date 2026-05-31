package toolemu

import "testing"

func TestDefaultsApplied(t *testing.T) {
	got := ToolEmulationConfig{}.DefaultsApplied()
	if got.OnParseFailure != "parse_failed_to_content" {
		t.Fatalf("OnParseFailure default = %q, want parse_failed_to_content", got.OnParseFailure)
	}
}

func TestDefaultsApplied_PreservesExplicit(t *testing.T) {
	in := ToolEmulationConfig{OnParseFailure: "error"}
	got := in.DefaultsApplied()
	if got.OnParseFailure != "error" {
		t.Fatalf("OnParseFailure = %q, want error", got.OnParseFailure)
	}
}

func TestToolEmulationConfigDefaultsDoNotRequirePromptTemplate(t *testing.T) {
	cfg := ToolEmulationConfig{Enabled: true}
	out := cfg.DefaultsApplied()
	if out.OnParseFailure != "parse_failed_to_content" {
		t.Fatalf("OnParseFailure = %q", out.OnParseFailure)
	}
	if !out.Enabled {
		t.Fatal("Enabled should be preserved")
	}
}
