package toolemu

import (
	"encoding/json"
	"fmt"
)

// FakeStreamResponses slices a Parsed result into an OpenAI Responses event
// stream. Each chunk is one "event: <type>\ndata: <json>\n\n" frame. The body
// JSON always carries a top-level "type" field that mirrors the event name so
// downstream consumers can route off either signal.
func FakeStreamResponses(p Parsed, meta UpstreamMeta) ([][]byte, error) {
	var out [][]byte
	respStub := map[string]any{
		"id": meta.ResponseID, "object": "response",
		"created_at": meta.Created, "model": meta.Model,
		"status": "in_progress",
	}
	out = append(out, sseEvent("response.created", map[string]any{"type": "response.created", "response": respStub}))
	out = append(out, sseEvent("response.in_progress", map[string]any{"type": "response.in_progress", "response": respStub}))

	outIndex := 0
	if p.Prose != "" {
		itemID := "msg_" + meta.ResponseID
		msgItem := map[string]any{
			"type": "message", "role": "assistant", "id": itemID, "status": "in_progress",
			"content": []any{},
		}
		out = append(out, sseEvent("response.output_item.added", map[string]any{
			"type":         "response.output_item.added",
			"output_index": outIndex, "item": msgItem,
		}))
		out = append(out, sseEvent("response.content_part.added", map[string]any{
			"type":         "response.content_part.added",
			"output_index": outIndex, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": ""},
		}))
		for _, chunk := range runeChunks(p.Prose, proseChunkSize) {
			out = append(out, sseEvent("response.output_text.delta", map[string]any{
				"type":         "response.output_text.delta",
				"output_index": outIndex, "content_index": 0,
				"delta": chunk,
			}))
		}
		out = append(out, sseEvent("response.output_text.done", map[string]any{
			"type":         "response.output_text.done",
			"output_index": outIndex, "content_index": 0, "text": p.Prose,
		}))
		out = append(out, sseEvent("response.content_part.done", map[string]any{
			"type":         "response.content_part.done",
			"output_index": outIndex, "content_index": 0,
			"part": map[string]any{"type": "output_text", "text": p.Prose},
		}))
		out = append(out, sseEvent("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": outIndex,
			"item": map[string]any{
				"type": "message", "role": "assistant", "id": itemID, "status": "completed",
				"content": []any{map[string]any{"type": "output_text", "text": p.Prose}},
			},
		}))
		outIndex++
	}

	for i, c := range p.ToolCalls {
		id := c.ID
		if id == "" {
			id = DeriveID(meta, i)
		}
		args := string(c.Arguments)
		callItem := map[string]any{
			"type": "function_call", "id": id, "call_id": id,
			"name": c.Name, "arguments": "", "status": "in_progress",
		}
		out = append(out, sseEvent("response.output_item.added", map[string]any{
			"type":         "response.output_item.added",
			"output_index": outIndex, "item": callItem,
		}))
		for _, chunk := range runeChunks(args, argChunkSize) {
			out = append(out, sseEvent("response.function_call_arguments.delta", map[string]any{
				"type":         "response.function_call_arguments.delta",
				"output_index": outIndex, "item_id": id, "delta": chunk,
			}))
		}
		out = append(out, sseEvent("response.function_call_arguments.done", map[string]any{
			"type":         "response.function_call_arguments.done",
			"output_index": outIndex, "item_id": id, "arguments": args,
		}))
		out = append(out, sseEvent("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"output_index": outIndex,
			"item": map[string]any{
				"type": "function_call", "id": id, "call_id": id,
				"name": c.Name, "arguments": args, "status": "completed",
			},
		}))
		outIndex++
	}

	completed := map[string]any{
		"id": meta.ResponseID, "object": "response",
		"created_at": meta.Created, "model": meta.Model,
		"status": "completed",
	}
	if len(meta.UsagePayload) > 0 {
		var u any
		if err := json.Unmarshal(meta.UsagePayload, &u); err == nil {
			completed["usage"] = u
		}
	}
	out = append(out, sseEvent("response.completed", map[string]any{"type": "response.completed", "response": completed}))
	return out, nil
}

func sseEvent(eventType string, body map[string]any) []byte {
	raw, _ := json.Marshal(body)
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, raw))
}
