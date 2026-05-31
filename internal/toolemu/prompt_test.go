package toolemu

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRenderInjection_Stable(t *testing.T) {
	tools := []ToolSpec{
		{Name: "get_weather", Description: "city weather",
			SchemaJSON: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
		{Name: "search_web", Description: "web search",
			SchemaJSON: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)},
	}
	a := RenderInjection(tools, ToolChoiceAuto)
	b := RenderInjection(tools, ToolChoiceAuto)
	if a != b {
		t.Fatal("render must be deterministic for same input")
	}
}

func TestRenderInjection_ReorderInvariant(t *testing.T) {
	a := RenderInjection([]ToolSpec{
		{Name: "alpha", Description: "a", SchemaJSON: json.RawMessage(`{}`)},
		{Name: "beta", Description: "b", SchemaJSON: json.RawMessage(`{}`)},
	}, ToolChoiceAuto)
	b := RenderInjection([]ToolSpec{
		{Name: "beta", Description: "b", SchemaJSON: json.RawMessage(`{}`)},
		{Name: "alpha", Description: "a", SchemaJSON: json.RawMessage(`{}`)},
	}, ToolChoiceAuto)
	if a != b {
		t.Fatal("tool order must not affect rendered text")
	}
}

func TestRenderInjection_SchemaKeySortStable(t *testing.T) {
	a := RenderInjection([]ToolSpec{{
		Name: "x", Description: "d",
		SchemaJSON: json.RawMessage(`{"a":1,"b":2}`),
	}}, ToolChoiceAuto)
	b := RenderInjection([]ToolSpec{{
		Name: "x", Description: "d",
		SchemaJSON: json.RawMessage(`{"b":2,"a":1}`),
	}}, ToolChoiceAuto)
	if a != b {
		t.Fatalf("schema key order must not affect rendering\nA:\n%s\nB:\n%s", a, b)
	}
}

func TestRenderInjection_RequiresToolCallName(t *testing.T) {
	out := RenderInjection([]ToolSpec{{Name: "x", Description: "d", SchemaJSON: json.RawMessage(`{}`)}}, ToolChoiceAuto)
	if !strings.Contains(out, `"name" field is REQUIRED`) {
		t.Fatalf("tool protocol must state name is required:\n%s", out)
	}
}

// TestRenderInjection_RejectsNativeToolCallEnvelopes verifies the protocol
// text explicitly forbids common provider-native tool-call envelopes. Kimi K2
// / Qwen3-style models otherwise revert to their training-time
// <|tool_call_begin|> sentinels, which our parser will not recognize.
func TestRenderInjection_RejectsNativeToolCallEnvelopes(t *testing.T) {
	out := RenderInjection([]ToolSpec{{Name: "x", Description: "d", SchemaJSON: json.RawMessage(`{}`)}}, ToolChoiceAuto)
	mustContain := []string{
		"<|tool_call_begin|>",
		"<|tool_calls_section_begin|>",
		`do NOT prefix it ("functions."`,
		`do NOT suffix it with an index (":0"`,
		"Markdown-fenced JSON",
	}
	for _, needle := range mustContain {
		if !strings.Contains(out, needle) {
			t.Fatalf("tool protocol missing %q in:\n%s", needle, out)
		}
	}
}

func TestRenderInjection_ToolChoiceClauses(t *testing.T) {
	tools := []ToolSpec{{Name: "x", Description: "d", SchemaJSON: json.RawMessage(`{}`)}}
	cases := []struct {
		tc     ToolChoice
		needle string
	}{
		{ToolChoiceAuto, "You MAY call a tool"},
		{ToolChoiceNone, "You MUST NOT call any tool"},
		{ToolChoiceRequired, "You MUST call at least one tool"},
		{ToolChoiceNamed("foo"), `You MUST call the tool named "foo"`},
	}
	for _, c := range cases {
		out := RenderInjection(tools, c.tc)
		if !strings.Contains(out, c.needle) {
			t.Fatalf("clause %v missing %q in:\n%s", c.tc, c.needle, out)
		}
	}
}

func TestRenderInjection_ToolsDocUsesOpenAIChatToolsJSON(t *testing.T) {
	out := RenderInjection([]ToolSpec{{
		Name:        "search_web",
		Description: "search the web\nwith a second line that must stay inside JSON",
		SchemaJSON:  json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
	}}, ToolChoiceAuto)

	start := strings.Index(out, "<tools_doc>\n")
	end := strings.Index(out, "\n</tools_doc>")
	if start < 0 || end < 0 || end <= start {
		t.Fatalf("missing tools_doc block:\n%s", out)
	}
	doc := out[start+len("<tools_doc>\n") : end]

	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal([]byte(doc), &tools); err != nil {
		t.Fatalf("tools_doc must be JSON: %v\n%s", err, doc)
	}
	if len(tools) != 1 {
		t.Fatalf("expected one tool, got %d: %s", len(tools), doc)
	}
	if tools[0].Type != "function" || tools[0].Function.Name != "search_web" {
		t.Fatalf("unexpected OpenAI tool shape: %+v", tools[0])
	}
	if tools[0].Function.Description != "search the web\nwith a second line that must stay inside JSON" {
		t.Fatalf("description was not preserved as JSON string: %q", tools[0].Function.Description)
	}
	if got := string(tools[0].Function.Parameters); got != `{"properties":{"q":{"type":"string"}},"required":["q"],"type":"object"}` {
		t.Fatalf("parameters must be canonical JSON, got %s", got)
	}
}

func TestRenderInjection_ContainsBothBlocks(t *testing.T) {
	out := RenderInjection([]ToolSpec{{Name: "x", Description: "d", SchemaJSON: json.RawMessage(`{}`)}}, ToolChoiceAuto)
	if !strings.Contains(out, "<tools_doc>") || !strings.Contains(out, "</tools_doc>") {
		t.Fatalf("missing tools_doc block:\n%s", out)
	}
	if !strings.Contains(out, "<tool_protocol>") || !strings.Contains(out, "</tool_protocol>") {
		t.Fatalf("missing tool_protocol block:\n%s", out)
	}
}
