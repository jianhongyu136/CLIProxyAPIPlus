package test

import (
	"bytes"
	"context"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/toolemu"
	"github.com/tidwall/gjson"
)

// TestToolEmuResponsesRoundTrip exercises helps.RunToolEmu against a stub
// upstream returning an OpenAI Responses non-stream body that carries a
// <tool_call> sentinel inside output_text. The BuiltBody must surface a
// function_call output entry naming the parsed tool.
func TestToolEmuResponsesRoundTrip(t *testing.T) {
	payload := []byte(`{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"weather in sf"}]}],"tools":[{"type":"function","name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"loc":{"type":"string"}}}}]}`)
	upstream := []byte(`{"id":"resp_1","object":"response","created_at":1700000000,"model":"m","status":"completed","output":[{"type":"message","role":"assistant","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"loc\":\"sf\"}}</tool_call>"}]}],"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}`)
	send := func(ctx context.Context, body []byte) ([]byte, error) { return upstream, nil }
	outcome, err := helps.RunToolEmu(context.Background(), payload, toolemu.ShapeOpenAIResponses, "openai-compatibility", toolemu.RetryPolicy{}, send)
	if err != nil {
		t.Fatalf("RunToolEmu: %v", err)
	}
	if got := gjson.GetBytes(outcome.Folded, "input.0.content.0.text").String(); !bytes.Contains([]byte(got), []byte("<tool_protocol>")) {
		t.Fatalf("folded first user input part missing tool_protocol marker: %q", got)
	}
	if bytes.Contains(outcome.Folded, []byte(`"tools":[`)) {
		t.Fatalf("folded still contains tools array: %s", string(outcome.Folded))
	}
	fcName := gjson.GetBytes(outcome.BuiltBody, `output.#(type=="function_call").name`).String()
	if fcName != "get_weather" {
		t.Fatalf("expected function_call name get_weather, got %q (built=%s)", fcName, string(outcome.BuiltBody))
	}
	if len(outcome.FakeStreamChunks) == 0 {
		t.Fatalf("no fake-stream chunks")
	}
	// Each frame should start with the SSE event prefix.
	first := outcome.FakeStreamChunks[0]
	if !bytes.HasPrefix(first, []byte("event: response.created\n")) {
		t.Fatalf("first frame should be response.created event, got %s", string(first))
	}
	last := outcome.FakeStreamChunks[len(outcome.FakeStreamChunks)-1]
	if !bytes.HasPrefix(last, []byte("event: response.completed\n")) {
		t.Fatalf("last frame should be response.completed event, got %s", string(last))
	}
}
