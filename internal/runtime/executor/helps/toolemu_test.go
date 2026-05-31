package helps

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/toolemu"
	"github.com/tidwall/gjson"
)

func TestRunToolEmuBuildsM2Bodies(t *testing.T) {
	cases := []struct {
		name      string
		shape     toolemu.UpstreamShape
		payload   []byte
		body      []byte
		wantPath  string
		wantValue string
	}{
		{
			name:      "claude",
			shape:     toolemu.ShapeClaudeMessages,
			payload:   []byte(`{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":{}}]}`),
			body:      []byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"}]}`),
			wantPath:  "content.0.type",
			wantValue: "tool_use",
		},
		{
			name:      "gemini",
			shape:     toolemu.ShapeGeminiGenerateContent,
			payload:   []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"functionDeclarations":[{"name":"f","parameters":{}}]}]}`),
			body:      []byte(`{"modelVersion":"gemini-test","candidates":[{"content":{"parts":[{"text":"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"}]}}]}`),
			wantPath:  "candidates.0.content.parts.0.functionCall.name",
			wantValue: "f",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			send := func(_ context.Context, _ []byte) ([]byte, error) { return c.body, nil }
			outcome, err := RunToolEmu(context.Background(), c.payload, c.shape, "p", toolemu.RetryPolicy{}, send)
			if err != nil {
				t.Fatal(err)
			}
			if len(outcome.BuiltBody) == 0 {
				t.Fatal("BuiltBody must be populated")
			}
			if got := gjson.GetBytes(outcome.BuiltBody, c.wantPath).String(); got != c.wantValue {
				t.Fatalf("%s = %q, want %q: %s", c.wantPath, got, c.wantValue, outcome.BuiltBody)
			}
		})
	}
}

func TestRunToolEmuStreamUsesUpstreamMetadataAndUsage(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl_1","created":123,"model":"m1","choices":[{"delta":{"content":"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"},"index":0}]}`,
		``,
		`data: {"id":"chatcmpl_1","created":123,"model":"m1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	var frames [][]byte
	meta, err := RunToolEmuStream(context.Background(), toolemu.UpstreamMeta{Provider: "p"}, toolemu.ShapeOpenAIChat, strings.NewReader(sse), toolemu.ToolChoiceAuto, func(frame []byte) {
		frames = append(frames, bytes.Clone(frame))
	})
	if err != nil {
		t.Fatal(err)
	}
	if meta.ResponseID != "chatcmpl_1" || meta.Model != "m1" || meta.Created != 123 {
		t.Fatalf("meta = %+v", meta)
	}
	if got := gjson.GetBytes(meta.UsagePayload, "total_tokens").Int(); got != 5 {
		t.Fatalf("usage total_tokens = %d", got)
	}
	if len(frames) == 0 {
		t.Fatal("no frames emitted")
	}
	firstPayload := bytes.TrimPrefix(bytes.TrimSpace(frames[0]), []byte("data: "))
	if got := gjson.GetBytes(firstPayload, "id").String(); got != "chatcmpl_1" {
		t.Fatalf("frame id = %q", got)
	}
	if got := gjson.GetBytes(firstPayload, "created").Int(); got != 123 {
		t.Fatalf("frame created = %d", got)
	}
}
