package toolemu

import (
	"bytes"
	"strings"
	"testing"
)

func TestResponsesStreamEmitter_EventOrder(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewResponsesStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1", Created: 100}, emit)
	ev := em.Events()

	ev.OnProseDelta("Hi ")
	ev.OnProseDelta("there")
	ev.OnToolCallStart(0, "call_x", "f")
	ev.OnArgsDelta(0, `{"a":1}`)
	ev.OnToolCallEnd(0)
	ev.OnComplete()

	want := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	idx := 0
	for _, f := range frames {
		eventType, _ := responseEvent(f)
		if idx < len(want) && eventType == want[idx] {
			idx++
		}
	}
	if idx != len(want) {
		t.Fatalf("event order incomplete: matched %d/%d, frames=%d", idx, len(want), len(frames))
	}
}

func TestResponsesStreamEmitter_ReconstructProseAndArgs(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewResponsesStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1"}, emit)
	ev := em.Events()
	ev.OnProseDelta("Hello ")
	ev.OnProseDelta("world")
	ev.OnToolCallStart(0, "call_x", "f")
	ev.OnArgsDelta(0, `{"x":`)
	ev.OnArgsDelta(0, `1}`)
	ev.OnToolCallEnd(0)
	ev.OnComplete()

	var prose, args strings.Builder
	for _, f := range frames {
		eventType, body := responseEvent(f)
		switch eventType {
		case "response.output_text.delta":
			prose.WriteString(jsonStringField(body, "delta"))
		case "response.function_call_arguments.delta":
			args.WriteString(jsonStringField(body, "delta"))
		}
	}
	if prose.String() != "Hello world" {
		t.Fatalf("prose = %q", prose.String())
	}
	if args.String() != `{"x":1}` {
		t.Fatalf("args = %q", args.String())
	}
}

func TestResponsesStreamEmitter_ToolCallOnly_NoMessageItem(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewResponsesStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1"}, emit)
	ev := em.Events()
	ev.OnToolCallStart(0, "call_x", "f")
	ev.OnArgsDelta(0, `{}`)
	ev.OnToolCallEnd(0)
	ev.OnComplete()

	hasMessageItem := false
	for _, f := range frames {
		eventType, body := responseEvent(f)
		if eventType == "response.output_item.added" && bytes.Contains(body, []byte(`"type":"message"`)) {
			hasMessageItem = true
		}
	}
	if hasMessageItem {
		t.Fatal("should not emit message item when no prose")
	}
}

func TestResponsesStreamEmitter_ToolChoiceNamedRejectsWrongTool(t *testing.T) {
	var frames [][]byte
	emitter := NewResponsesStreamEmitter(UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r", Created: 1}, func(frame []byte) {
		frames = append(frames, append([]byte(nil), frame...))
	})
	emitter.SetToolChoice(ToolChoiceNamed("expected"))
	events := emitter.Events()
	events.OnToolCallStart(0, "call_1", "other")
	events.OnArgsDelta(0, `{}`)
	events.OnToolCallEnd(0)

	joined := string(bytes.Join(frames, nil))
	if strings.Contains(joined, "response.output_item.done") && strings.Contains(joined, `"function_call"`) {
		t.Fatalf("completed wrong tool call must not be emitted:\n%s", joined)
	}
	if !strings.Contains(joined, "tool_choice named requires only") {
		t.Fatalf("missing validation error event:\n%s", joined)
	}
}
