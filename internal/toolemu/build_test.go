package toolemu

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestBuildChatCompletion_ToolCallsOnly(t *testing.T) {
	p := Parsed{ToolCalls: []ParsedToolCall{{
		Name: "f", Arguments: []byte(`{"x":1}`),
	}}}
	meta := UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r"}
	out, err := BuildChatCompletion(p, meta)
	if err != nil {
		t.Fatal(err)
	}
	root := gjson.ParseBytes(out)
	if got := root.Get("choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("finish_reason = %q want tool_calls", got)
	}
	if got := root.Get("choices.0.message.tool_calls.0.function.arguments").String(); got != `{"x":1}` {
		t.Fatalf("arguments = %q", got)
	}
	id := root.Get("choices.0.message.tool_calls.0.id").String()
	if !strings.HasPrefix(id, "call_") {
		t.Fatalf("id missing call_ prefix: %q", id)
	}
	if root.Get("choices.0.message.content").Type != gjson.Null {
		t.Fatalf("content must be null when only tool_calls present: %s", out)
	}
}

func TestBuildChatCompletion_ProseOnly(t *testing.T) {
	p := Parsed{Prose: "hi"}
	out, err := BuildChatCompletion(p, UpstreamMeta{ResponseID: "r"})
	if err != nil {
		t.Fatal(err)
	}
	root := gjson.ParseBytes(out)
	if got := root.Get("choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("finish_reason = %q want stop", got)
	}
	if got := root.Get("choices.0.message.content").String(); got != "hi" {
		t.Fatalf("content = %q", got)
	}
}

func TestBuildChatCompletion_UsagePassthrough(t *testing.T) {
	meta := UpstreamMeta{
		ResponseID:   "r",
		UsagePayload: []byte(`{"prompt_tokens":1,"completion_tokens":2}`),
	}
	out, err := BuildChatCompletion(Parsed{Prose: "x"}, meta)
	if err != nil {
		t.Fatal(err)
	}
	if got := gjson.GetBytes(out, "usage.prompt_tokens").Int(); got != 1 {
		t.Fatalf("usage.prompt_tokens = %d", got)
	}
	if got := gjson.GetBytes(out, "usage.completion_tokens").Int(); got != 2 {
		t.Fatalf("usage.completion_tokens = %d", got)
	}
}

func TestBuildResponses_ProseAndCall(t *testing.T) {
	p := Parsed{
		Prose: "hello",
		ToolCalls: []ParsedToolCall{{
			Name: "f", Arguments: []byte(`{"a":1}`),
		}},
	}
	meta := UpstreamMeta{ResponseID: "r"}
	out, err := BuildResponses(p, meta)
	if err != nil {
		t.Fatal(err)
	}
	root := gjson.ParseBytes(out)
	if got := root.Get("output.0.type").String(); got != "message" {
		t.Fatalf("output[0].type = %q want message", got)
	}
	if got := root.Get("output.0.content.0.type").String(); got != "output_text" {
		t.Fatalf("output[0].content[0].type = %q want output_text", got)
	}
	if got := root.Get("output.1.type").String(); got != "function_call" {
		t.Fatalf("output[1].type = %q want function_call", got)
	}
	id := root.Get("output.1.id").String()
	callID := root.Get("output.1.call_id").String()
	if id == "" || id != callID {
		t.Fatalf("id (%q) must equal call_id (%q)", id, callID)
	}
	if got := root.Get("output.1.name").String(); got != "f" {
		t.Fatalf("output[1].name = %q", got)
	}
	if got := root.Get("output.1.arguments").String(); got != `{"a":1}` {
		t.Fatalf("output[1].arguments = %q", got)
	}
	if got := root.Get("output.1.status").String(); got != "completed" {
		t.Fatalf("output[1].status = %q", got)
	}
}

func TestBuildResponses_OnlyCall(t *testing.T) {
	p := Parsed{ToolCalls: []ParsedToolCall{{Name: "f", Arguments: []byte(`{}`)}}}
	out, err := BuildResponses(p, UpstreamMeta{ResponseID: "r"})
	if err != nil {
		t.Fatal(err)
	}
	root := gjson.ParseBytes(out)
	outputs := root.Get("output").Array()
	if len(outputs) != 1 {
		t.Fatalf("output len = %d want 1", len(outputs))
	}
	if got := outputs[0].Get("type").String(); got != "function_call" {
		t.Fatalf("output[0].type = %q want function_call", got)
	}
}

func TestBuildResponses_UsagePassthrough(t *testing.T) {
	meta := UpstreamMeta{
		ResponseID:   "r",
		UsagePayload: []byte(`{"input_tokens":4,"output_tokens":5}`),
	}
	out, err := BuildResponses(Parsed{Prose: "x"}, meta)
	if err != nil {
		t.Fatal(err)
	}
	if got := gjson.GetBytes(out, "usage.input_tokens").Int(); got != 4 {
		t.Fatalf("usage.input_tokens = %d", got)
	}
}
