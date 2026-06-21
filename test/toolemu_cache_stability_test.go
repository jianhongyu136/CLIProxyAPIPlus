package test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/toolemu"
	"github.com/tidwall/gjson"
)

func TestToolEmuFoldDeterministic(t *testing.T) {
	payload := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"loc":{"type":"string"}}}}}]}`)
	a, err := toolemu.FoldRequest(payload, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold a: %v", err)
	}
	b, err := toolemu.FoldRequest(payload, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold b: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("non-deterministic fold output:\nA=%s\nB=%s", string(a), string(b))
	}
}

func TestToolEmuFoldPrefixStableAcrossUserMessage(t *testing.T) {
	base := `{"model":"x","tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object"}}}]`
	p1 := []byte(base + `,"messages":[{"role":"user","content":"hello"}]}`)
	p2 := []byte(base + `,"messages":[{"role":"user","content":"a different question that is much longer"}]}`)
	a, err := toolemu.FoldRequest(p1, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold p1: %v", err)
	}
	b, err := toolemu.FoldRequest(p2, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold p2: %v", err)
	}
	// The injected prompt is prepended to the first user message; compare the
	// injected prefix across user-message variants to verify cache stability.
	sysA := gjson.GetBytes(a, "messages.0.content").String()
	sysB := gjson.GetBytes(b, "messages.0.content").String()
	if sysA == "" || sysB == "" {
		t.Fatalf("messages.0.content empty: A=%q B=%q", sysA, sysB)
	}
	prefixA := strings.TrimSuffix(sysA, "hello")
	prefixB := strings.TrimSuffix(sysB, "a different question that is much longer")
	if prefixA != prefixB {
		t.Fatalf("injected user prefix differs across user-message variants — prefix not cache-stable\nA=%q\nB=%q", prefixA, prefixB)
	}
	// Sanity: the injected content carries the tool_protocol sentinel.
	if !bytes.Contains([]byte(prefixA), []byte("<tool_protocol>")) {
		t.Fatalf("injected user prefix missing tool_protocol marker: %q", prefixA)
	}
}

func TestToolEmuFoldResponsesDeterministic(t *testing.T) {
	payload := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"loc":{"type":"string"}}}}]}`)
	a, err := toolemu.FoldRequest(payload, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIResponses, Provider: "p"})
	if err != nil {
		t.Fatalf("fold a: %v", err)
	}
	b, err := toolemu.FoldRequest(payload, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIResponses, Provider: "p"})
	if err != nil {
		t.Fatalf("fold b: %v", err)
	}
	if !bytes.Equal(a, b) {
		t.Fatalf("non-deterministic fold output:\nA=%s\nB=%s", string(a), string(b))
	}
	if got := gjson.GetBytes(a, "input.0.content.0.text").String(); !bytes.Contains([]byte(got), []byte("<tool_protocol>")) {
		t.Fatalf("first user input part missing tool_protocol marker: %q", got)
	}
}

// TestToolEmuFoldChatNoEscapeInWire ensures the chat fold path writes literal
// `<tool_protocol>` / `<tool_call>` sentinels into the folded wire bytes
// instead of HTML-escaped `<...>` sequences. Escape divergence
// between the "has system" and "no system" insertion paths fragments the
// upstream prefix cache, so we lock the no-escape contract here.
func TestToolEmuFoldChatLiteralSentinelsInWire(t *testing.T) {
	noSys := []byte(`{"model":"x","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}]}`)
	withSys := []byte(`{"model":"x","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}]}`)
	a, err := toolemu.FoldRequest(noSys, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold noSys: %v", err)
	}
	b, err := toolemu.FoldRequest(withSys, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIChat, Provider: "p"})
	if err != nil {
		t.Fatalf("fold withSys: %v", err)
	}
	for _, c := range []struct {
		name  string
		bytes []byte
	}{{"noSys", a}, {"withSys", b}} {
		if !bytes.Contains(c.bytes, []byte("<tool_protocol>")) {
			t.Fatalf("%s: folded wire bytes should carry literal <tool_protocol>, got: %s", c.name, string(c.bytes))
		}
		if bytes.Contains(c.bytes, []byte("\\u003ctool_protocol")) {
			t.Fatalf("%s: folded wire bytes must not HTML-escape `<` to \\u003c: %s", c.name, string(c.bytes))
		}
	}
}

// TestToolEmuFoldResponsesLiteralSentinelsInWire mirrors the chat sentinel
// check for the Responses shape — folded `instructions` must contain literal
// `<tool_protocol>` rather than `<tool_protocol>`. Regression guard
// for the same sjson HTML-escape fallback we hit on the chat path.
func TestToolEmuFoldResponsesLiteralSentinelsInWire(t *testing.T) {
	noInstr := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object"}}]}`)
	withInstr := []byte(`{"model":"m","instructions":"sys","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"get_weather","description":"w","parameters":{"type":"object"}}]}`)
	a, err := toolemu.FoldRequest(noInstr, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIResponses, Provider: "p"})
	if err != nil {
		t.Fatalf("fold noInstr: %v", err)
	}
	b, err := toolemu.FoldRequest(withInstr, toolemu.FoldOpts{Shape: toolemu.ShapeOpenAIResponses, Provider: "p"})
	if err != nil {
		t.Fatalf("fold withInstr: %v", err)
	}
	for _, c := range []struct {
		name  string
		bytes []byte
	}{{"noInstr", a}, {"withInstr", b}} {
		if !bytes.Contains(c.bytes, []byte("<tool_protocol>")) {
			t.Fatalf("%s: folded wire bytes should carry literal <tool_protocol>, got: %s", c.name, string(c.bytes))
		}
		if bytes.Contains(c.bytes, []byte("\\u003ctool_protocol")) {
			t.Fatalf("%s: folded wire bytes must not HTML-escape `<` to \\u003c: %s", c.name, string(c.bytes))
		}
	}
}
