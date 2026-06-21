package test

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/toolemu"
	"github.com/tidwall/gjson"
)

// TestToolEmuChatRoundTrip exercises the helps.RunToolEmu glue against a stub
// upstream that returns a chat-completions body containing a <tool_call> block.
// The resulting BuiltBody must reshape that into a native tool_calls choice
// with finish_reason=="tool_calls".
func TestToolEmuChatRoundTrip(t *testing.T) {
	payload := []byte(`{"model":"m","messages":[{"role":"user","content":"weather in sf"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"loc":{"type":"string"}},"required":["loc"]}}}]}`)
	upstreamBody := []byte(`{"id":"chatcmpl-1","object":"chat.completion","created":1700000000,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"Calling tool now.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"loc\":\"sf\"}}</tool_call>"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}`)
	send := func(ctx context.Context, body []byte) ([]byte, error) {
		return upstreamBody, nil
	}
	outcome, err := helps.RunToolEmu(context.Background(), payload, toolemu.ShapeOpenAIChat, "openai-compatibility", toolemu.RetryPolicy{}, send)
	if err != nil {
		t.Fatalf("RunToolEmu: %v", err)
	}
	if got := gjson.GetBytes(outcome.Folded, "messages.0.content").String(); !strings.Contains(got, "<tool_protocol>") {
		t.Fatalf("folded first user content missing tool_protocol marker: %q", got)
	}
	if bytes.Contains(outcome.Folded, []byte(`"tools":[`)) {
		t.Fatalf("folded still contains tools array: %s", string(outcome.Folded))
	}
	name := gjson.GetBytes(outcome.BuiltBody, "choices.0.message.tool_calls.0.function.name").String()
	if name != "get_weather" {
		t.Fatalf("expected tool name get_weather, got %q (built=%s)", name, string(outcome.BuiltBody))
	}
	if got := gjson.GetBytes(outcome.BuiltBody, "choices.0.finish_reason").String(); got != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %q (built=%s)", got, string(outcome.BuiltBody))
	}
	if len(outcome.FakeStreamChunks) == 0 {
		t.Fatalf("no fake-stream chunks")
	}
	var built map[string]any
	if err := json.Unmarshal(outcome.BuiltBody, &built); err != nil {
		t.Fatalf("unmarshal built: %v", err)
	}
	if _, ok := built["usage"]; !ok {
		t.Fatalf("usage missing from built body: %s", string(outcome.BuiltBody))
	}
	args := gjson.GetBytes(outcome.BuiltBody, "choices.0.message.tool_calls.0.function.arguments").String()
	if !strings.Contains(args, `"loc":"sf"`) {
		t.Fatalf("unexpected arguments: %s", args)
	}
}

// TestToolEmuChatDegradeToContent verifies that when the upstream never emits
// a valid tool_call, the soft-retry exhausts and the result degrades to prose
// content (the default failure policy).
func TestToolEmuChatDegradeToContent(t *testing.T) {
	payload := []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object"}}}]}`)
	bad := []byte(`{"id":"chatcmpl-x","object":"chat.completion","created":1700000000,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"<tool_call>not json</tool_call>"},"finish_reason":"stop"}]}`)
	attempts := 0
	send := func(ctx context.Context, body []byte) ([]byte, error) {
		attempts++
		return bad, nil
	}
	outcome, err := helps.RunToolEmu(context.Background(), payload, toolemu.ShapeOpenAIChat, "openai-compatibility", toolemu.RetryPolicy{Attempts: 1, OnFailure: "parse_failed_to_content"}, send)
	if err != nil {
		t.Fatalf("RunToolEmu: %v", err)
	}
	if !outcome.Result.Degraded {
		t.Fatalf("expected Degraded=true on parse failure")
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts (initial + 1 retry), got %d", attempts)
	}
	content := gjson.GetBytes(outcome.BuiltBody, "choices.0.message.content").String()
	if !strings.Contains(content, "<tool_call>not json</tool_call>") {
		t.Fatalf("expected degraded content to surface raw text, got %q", content)
	}
}
