package toolemu

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/tidwall/gjson"
)

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestFoldChatHistory_AssistantWithToolCalls(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"role": "assistant", "content": "Let me check.", "tool_calls": []any{
			map[string]any{
				"id":   "call_1",
				"type": "function",
				"function": map[string]any{
					"name":      "get_weather",
					"arguments": `{"city":"SF"}`,
				},
			},
		}},
	})
	out, err := FoldChatHistory(in)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`<tool_call>`)) || !bytes.Contains(out, []byte(`get_weather`)) {
		t.Fatalf("expected folded tool_call in:\n%s", out)
	}
	if bytes.Contains(out, []byte(`"tool_calls"`)) {
		t.Fatalf("original tool_calls field must be removed:\n%s", out)
	}
}

func TestFoldChatHistory_ToolResultBecomesUserMessage(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"role": "assistant", "content": "", "tool_calls": []any{
			map[string]any{"id": "call_a", "type": "function",
				"function": map[string]any{"name": "f", "arguments": "{}"}},
		}},
		map[string]any{"role": "tool", "tool_call_id": "call_a", "content": `{"ok":true}`},
	})
	out, err := FoldChatHistory(in)
	if err != nil {
		t.Fatal(err)
	}
	content := gjson.GetBytes(out, "1.content").String()
	if content != `<tool_result index="0">{"ok":true}</tool_result>` {
		t.Fatalf("missing tool_result block:\n%s", out)
	}
	if bytes.Contains(out, []byte(`call_a`)) {
		t.Fatalf("folded prompt must not contain volatile tool id:\n%s", out)
	}
	if bytes.Contains(out, []byte(`"role":"tool"`)) || bytes.Contains(out, []byte(`"role": "tool"`)) {
		t.Fatalf("role=tool message must be replaced with role=user:\n%s", out)
	}
}

func TestFoldChatHistory_PreservesAssistantArrayContent(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "intro"},
			map[string]any{"type": "text", "text": "details"},
		}, "tool_calls": []any{
			map[string]any{"id": "call_a", "type": "function",
				"function": map[string]any{"name": "f", "arguments": "{}"}},
		}},
	})
	out, err := FoldChatHistory(in)
	if err != nil {
		t.Fatal(err)
	}
	content := gjson.GetBytes(out, "0.content")
	if !content.IsArray() {
		t.Fatalf("assistant content must stay as an array:\n%s", out)
	}
	parts := content.Array()
	if len(parts) != 3 {
		t.Fatalf("expected 3 content parts, got %d:\n%s", len(parts), out)
	}
	if parts[0].Get("text").String() != "intro" || parts[1].Get("text").String() != "details" {
		t.Fatalf("original assistant parts must be preserved:\n%s", out)
	}
	if parts[2].Get("type").String() != "text" || !bytes.Contains([]byte(parts[2].Get("text").String()), []byte(`<tool_call>`)) {
		t.Fatalf("folded tool_call must be appended as a new text part:\n%s", out)
	}
}

func TestFoldChatHistory_ArgumentsKeyOrderStable(t *testing.T) {
	a, _ := FoldChatHistory(mustJSON(t, []any{
		map[string]any{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "c", "type": "function",
				"function": map[string]any{"name": "n", "arguments": `{"a":1,"b":2}`}},
		}},
	}))
	b, _ := FoldChatHistory(mustJSON(t, []any{
		map[string]any{"role": "assistant", "tool_calls": []any{
			map[string]any{"id": "c", "type": "function",
				"function": map[string]any{"name": "n", "arguments": `{"b":2,"a":1}`}},
		}},
	}))
	if !bytes.Equal(a, b) {
		t.Fatalf("arguments key order must not affect fold output\nA: %s\nB: %s", a, b)
	}
}

func TestFoldChatHistory_RoundTripStable(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"role": "user", "content": "hi"},
		map[string]any{"role": "assistant", "content": "ok"},
	})
	a, _ := FoldChatHistory(in)
	b, _ := FoldChatHistory(in)
	if !bytes.Equal(a, b) {
		t.Fatal("fold must be deterministic")
	}
}
