package toolemu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
)

// FoldChatHistory rewrites a chat-completions messages JSON array, folding
// assistant.tool_calls and role=tool messages into <tool_call>/<tool_result>
// text blocks. The output is byte-stable for logically equal inputs (R3).
func FoldChatHistory(messages []byte) ([]byte, error) {
	if !gjson.ValidBytes(messages) || !gjson.ParseBytes(messages).IsArray() {
		return nil, fmt.Errorf("toolemu: messages payload is not a JSON array")
	}

	var raw []json.RawMessage
	if err := json.Unmarshal(messages, &raw); err != nil {
		return nil, err
	}

	out := make([]json.RawMessage, 0, len(raw))
	i := 0
	for i < len(raw) {
		role := gjson.GetBytes(raw[i], "role").String()
		if role == "assistant" {
			folded, j, err := foldAssistant(raw, i)
			if err != nil {
				return nil, err
			}
			out = append(out, folded...)
			i = j
			continue
		}
		if role == "tool" {
			// orphan tool message (no preceding assistant.tool_calls); still convert to user
			block, j, err := foldOrphanTools(raw, i)
			if err != nil {
				return nil, err
			}
			out = append(out, block)
			i = j
			continue
		}
		out = append(out, raw[i])
		i++
	}
	return marshalSorted(out)
}

// foldAssistant rewrites assistant[i] and consumes trailing role=tool messages.
// Returns the folded slice (one or two messages) and the next index to process.
func foldAssistant(raw []json.RawMessage, i int) ([]json.RawMessage, int, error) {
	asst := raw[i]
	toolCalls := gjson.GetBytes(asst, "tool_calls")
	if !toolCalls.Exists() || !toolCalls.IsArray() {
		return []json.RawMessage{asst}, i + 1, nil
	}

	// Collect original assistant content so array-based multimodal content stays
	// intact after we append the folded tool_call text.
	var (
		contentText  string
		contentParts []any
		contentIsArr bool
	)
	if c := gjson.GetBytes(asst, "content"); c.Exists() {
		switch {
		case c.IsArray():
			if err := json.Unmarshal([]byte(c.Raw), &contentParts); err != nil {
				return nil, 0, err
			}
			contentIsArr = true
		case c.Type == gjson.String:
			contentText = c.String()
		}
	}

	// Build folded assistant content: prior text + <tool_call> blocks (in array order).
	var sb strings.Builder
	sb.WriteString(contentText)
	type toolCallInfo struct {
		id    string
		index int
	}
	var calls []toolCallInfo
	callIndex := 0
	toolCalls.ForEach(func(_, tc gjson.Result) bool {
		id := tc.Get("id").String()
		name := tc.Get("function.name").String()
		argsRaw := tc.Get("function.arguments").String()
		argsCanonical := canonicalArgs(argsRaw)
		obj := map[string]any{"name": name, "arguments": json.RawMessage(argsCanonical), "index": callIndex}
		stable, _ := marshalSorted(obj)
		if sb.Len() > 0 && !strings.HasSuffix(sb.String(), "\n") {
			sb.WriteString("\n")
		}
		sb.WriteString("<tool_call>\n")
		sb.Write(stable)
		sb.WriteString("\n</tool_call>")
		calls = append(calls, toolCallInfo{id: id, index: callIndex})
		callIndex++
		return true
	})

	var content any
	if contentIsArr {
		content = append(contentParts, map[string]any{"type": "text", "text": sb.String()})
	} else {
		content = sb.String()
	}

	rebuilt, err := rebuildAssistant(asst, content)
	if err != nil {
		return nil, 0, err
	}

	// Consume contiguous role=tool messages that reference these tool_call ids.
	j := i + 1
	resultsByID := map[string]string{}
	for j < len(raw) && gjson.GetBytes(raw[j], "role").String() == "tool" {
		id := gjson.GetBytes(raw[j], "tool_call_id").String()
		content := gjson.GetBytes(raw[j], "content").String()
		resultsByID[id] = content
		j++
	}
	if len(resultsByID) == 0 {
		return []json.RawMessage{rebuilt}, j, nil
	}

	var rb strings.Builder
	for _, c := range calls {
		if v, ok := resultsByID[c.id]; ok {
			fmt.Fprintf(&rb, "<tool_result index=%q>%s</tool_result>\n", fmt.Sprintf("%d", c.index), v)
		}
	}
	userMsg, _ := marshalSorted(map[string]any{
		"role":    "user",
		"content": strings.TrimRight(rb.String(), "\n"),
	})
	return []json.RawMessage{rebuilt, userMsg}, j, nil
}

func foldOrphanTools(raw []json.RawMessage, i int) (json.RawMessage, int, error) {
	var rb strings.Builder
	j := i
	resultIndex := 0
	for j < len(raw) && gjson.GetBytes(raw[j], "role").String() == "tool" {
		content := gjson.GetBytes(raw[j], "content").String()
		fmt.Fprintf(&rb, "<tool_result index=%q>%s</tool_result>\n", fmt.Sprintf("%d", resultIndex), content)
		resultIndex++
		j++
	}
	out, _ := marshalSorted(map[string]any{
		"role":    "user",
		"content": strings.TrimRight(rb.String(), "\n"),
	})
	return out, j, nil
}

func rebuildAssistant(orig json.RawMessage, newContent any) (json.RawMessage, error) {
	var obj map[string]any
	if err := json.Unmarshal(orig, &obj); err != nil {
		return nil, err
	}
	delete(obj, "tool_calls")
	obj["content"] = newContent
	return marshalSorted(obj)
}

// canonicalArgs normalizes an arguments string into canonical sorted JSON.
// OpenAI sends arguments as a string-encoded JSON; we decode and re-encode.
func canonicalArgs(raw string) []byte {
	trim := bytes.TrimSpace([]byte(raw))
	if len(trim) == 0 {
		return []byte("{}")
	}
	var v any
	if err := json.Unmarshal(trim, &v); err != nil {
		return []byte("{}")
	}
	out, err := marshalSorted(v)
	if err != nil {
		return []byte("{}")
	}
	return out
}
