package toolemu

import (
	"bytes"
	"testing"

	"github.com/tidwall/gjson"
)

func TestFoldResponsesInput_FunctionCallFolded(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"type": "message", "role": "user",
			"content": []any{map[string]any{"type": "input_text", "text": "hi"}}},
		map[string]any{"type": "function_call", "id": "call_x", "name": "search",
			"arguments": `{"q":"foo"}`},
	})
	out, err := FoldResponsesInput(in)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(out, []byte(`<tool_call>`)) {
		t.Fatalf("missing tool_call:\n%s", out)
	}
	if bytes.Contains(out, []byte(`"function_call"`)) {
		t.Fatalf("function_call item must be removed:\n%s", out)
	}
}

func TestFoldResponsesInput_FunctionCallOutputFolded(t *testing.T) {
	in := mustJSON(t, []any{
		map[string]any{"type": "function_call", "id": "call_x", "name": "f", "arguments": "{}"},
		map[string]any{"type": "function_call_output", "call_id": "call_x", "output": `{"ok":true}`},
	})
	out, err := FoldResponsesInput(in)
	if err != nil {
		t.Fatal(err)
	}
	content := gjson.GetBytes(out, "1.content.0.text").String()
	if content != `<tool_result index="0">{"ok":true}</tool_result>` {
		t.Fatalf("missing tool_result block:\n%s", out)
	}
	if bytes.Contains(out, []byte(`call_x`)) {
		t.Fatalf("folded prompt must not contain volatile tool id:\n%s", out)
	}
}

func TestFoldResponsesInput_Deterministic(t *testing.T) {
	a, _ := FoldResponsesInput(mustJSON(t, []any{
		map[string]any{"type": "function_call", "id": "c", "name": "n",
			"arguments": `{"a":1,"b":2}`},
	}))
	b, _ := FoldResponsesInput(mustJSON(t, []any{
		map[string]any{"type": "function_call", "id": "c", "name": "n",
			"arguments": `{"b":2,"a":1}`},
	}))
	if !bytes.Equal(a, b) {
		t.Fatalf("arguments key order should not change output\nA: %s\nB: %s", a, b)
	}
}
