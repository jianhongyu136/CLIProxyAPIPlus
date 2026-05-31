package toolemu

import (
	"encoding/json"
	"fmt"
)

const (
	proseChunkSize = 80
	argChunkSize   = 120
)

// FakeStreamChat slices a Parsed result into a sequence of OpenAI
// chat-completion SSE chunks (each chunk is one `data: {...}\n\n` line plus a
// trailing `data: [DONE]\n\n`).
func FakeStreamChat(p Parsed, meta UpstreamMeta) ([][]byte, error) {
	base := map[string]any{
		"id":      meta.ResponseID,
		"object":  "chat.completion.chunk",
		"created": meta.Created,
		"model":   meta.Model,
	}
	var out [][]byte

	out = append(out, sseChatChunk(base, map[string]any{"role": "assistant"}, ""))

	for _, chunk := range runeChunks(p.Prose, proseChunkSize) {
		out = append(out, sseChatChunk(base, map[string]any{"content": chunk}, ""))
	}

	for i, c := range p.ToolCalls {
		id := c.ID
		if id == "" {
			id = DeriveID(meta, i)
		}
		header := []any{map[string]any{
			"index": i, "id": id, "type": "function",
			"function": map[string]any{"name": c.Name, "arguments": ""},
		}}
		out = append(out, sseChatChunk(base, map[string]any{"tool_calls": header}, ""))

		for _, chunk := range runeChunks(string(c.Arguments), argChunkSize) {
			delta := []any{map[string]any{
				"index":    i,
				"function": map[string]any{"arguments": chunk},
			}}
			out = append(out, sseChatChunk(base, map[string]any{"tool_calls": delta}, ""))
		}
	}

	finish := "stop"
	if len(p.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	out = append(out, sseChatChunk(base, map[string]any{}, finish))
	if len(meta.UsagePayload) > 0 {
		var usage any
		if err := json.Unmarshal(meta.UsagePayload, &usage); err == nil {
			out = append(out, sseChatUsageChunk(base, usage))
		}
	}
	out = append(out, []byte("data: [DONE]\n\n"))
	return out, nil
}

// runeChunks splits s into substrings of at most n runes each, preserving
// UTF-8 rune boundaries so no multibyte character is split across chunks.
func runeChunks(s string, n int) []string {
	if s == "" {
		return nil
	}
	var chunks []string
	for len(s) > 0 {
		count := 0
		end := len(s)
		for i := range s {
			if count == n {
				end = i
				break
			}
			count++
		}
		chunks = append(chunks, s[:end])
		s = s[end:]
	}
	return chunks
}

func sseChatChunk(base map[string]any, delta map[string]any, finish string) []byte {
	choice := map[string]any{"index": 0, "delta": delta}
	if finish != "" {
		choice["finish_reason"] = finish
	} else {
		choice["finish_reason"] = nil
	}
	frame := map[string]any{}
	for k, v := range base {
		frame[k] = v
	}
	frame["choices"] = []any{choice}
	raw, _ := json.Marshal(frame)
	return []byte(fmt.Sprintf("data: %s\n\n", raw))
}

func sseChatUsageChunk(base map[string]any, usage any) []byte {
	frame := map[string]any{}
	for k, v := range base {
		frame[k] = v
	}
	frame["choices"] = []any{}
	frame["usage"] = usage
	raw, _ := json.Marshal(frame)
	return []byte(fmt.Sprintf("data: %s\n\n", raw))
}
