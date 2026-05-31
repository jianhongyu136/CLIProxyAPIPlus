package toolemu

import "testing"

func TestIsEnabled_MasterSwitchOff(t *testing.T) {
	m := NewMatcher(ToolEmulationConfig{Enabled: false, Rules: []ToolEmulationRule{{Provider: "p"}}})
	if m.IsEnabled("p", "any-model", "") {
		t.Fatal("master switch off should disable all")
	}
}

func TestIsEnabled_ProviderOnly_AllModels(t *testing.T) {
	m := NewMatcher(ToolEmulationConfig{
		Enabled: true,
		Rules:   []ToolEmulationRule{{Provider: "openrouter"}},
	})
	if !m.IsEnabled("openrouter", "any", "") {
		t.Fatal("empty models+aliases should match all under provider")
	}
	if m.IsEnabled("openai", "any", "") {
		t.Fatal("different provider must not match")
	}
}

func TestIsEnabled_ModelGlob(t *testing.T) {
	m := NewMatcher(ToolEmulationConfig{
		Enabled: true,
		Rules: []ToolEmulationRule{{
			Provider: "openrouter",
			Models:   []string{"qwen/qwen3-*"},
		}},
	})
	if !m.IsEnabled("openrouter", "qwen/qwen3-235b", "") {
		t.Fatal("glob should match")
	}
	if m.IsEnabled("openrouter", "moonshot/k2", "") {
		t.Fatal("non-glob-matching model must not match")
	}
}

func TestIsEnabled_AliasMatch(t *testing.T) {
	m := NewMatcher(ToolEmulationConfig{
		Enabled: true,
		Rules: []ToolEmulationRule{{
			Provider:     "openai-compatible",
			ModelAliases: []string{"legacy-no-tools"},
		}},
	})
	if !m.IsEnabled("openai-compatible", "real-model-x", "legacy-no-tools") {
		t.Fatal("alias match should fire")
	}
	if m.IsEnabled("openai-compatible", "real-model-x", "other-alias") {
		t.Fatal("non-matching alias should not fire")
	}
}

func TestIsEnabled_FirstRuleWins(t *testing.T) {
	m := NewMatcher(ToolEmulationConfig{
		Enabled: true,
		Rules: []ToolEmulationRule{
			{Provider: "openrouter", Models: []string{"gpt-*"}},
			{Provider: "openrouter"},
		},
	})
	if !m.IsEnabled("openrouter", "gpt-4", "") {
		t.Fatal("first rule should match")
	}
	if !m.IsEnabled("openrouter", "anything", "") {
		t.Fatal("second rule should still apply for non-gpt model")
	}
}

func TestDefault_ReplaceTakesEffect(t *testing.T) {
	if Default.IsEnabled("p", "m", "") {
		t.Fatal("default matcher should start disabled")
	}
	Default.Replace(ToolEmulationConfig{Enabled: true, Rules: []ToolEmulationRule{{Provider: "p"}}})
	defer Default.Replace(ToolEmulationConfig{})
	if !Default.IsEnabled("p", "m", "") {
		t.Fatal("Replace did not take effect")
	}
}
