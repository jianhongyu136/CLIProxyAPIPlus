package toolemu

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestChatStreamEmitter_ProseAndToolCall(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewChatStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1", Created: 100}, emit)
	ev := em.Events()

	ev.OnProseDelta("Hello")
	ev.OnProseDelta(" world")
	ev.OnToolCallStart(0, "call_abc", "get_weather")
	ev.OnArgsDelta(0, `{"city":`)
	ev.OnArgsDelta(0, `"NYC"}`)
	ev.OnToolCallEnd(0)
	ev.OnComplete()

	if len(frames) == 0 {
		t.Fatal("no frames emitted")
	}
	if string(frames[len(frames)-1]) != "data: [DONE]\n\n" {
		t.Fatalf("last frame not DONE: %q", frames[len(frames)-1])
	}
	first := jsonPart(frames[0])
	if got := gjson.GetBytes(first, "choices.0.delta.role").String(); got != "assistant" {
		t.Fatalf("first frame role = %q want assistant", got)
	}

	var prose, args strings.Builder
	finishReason := ""
	for _, f := range frames {
		if bytes.Equal(f, []byte("data: [DONE]\n\n")) {
			continue
		}
		payload := jsonPart(f)
		if c := gjson.GetBytes(payload, "choices.0.delta.content"); c.Exists() {
			prose.WriteString(c.String())
		}
		if a := gjson.GetBytes(payload, "choices.0.delta.tool_calls.0.function.arguments"); a.Exists() {
			args.WriteString(a.String())
		}
		if fr := gjson.GetBytes(payload, "choices.0.finish_reason"); fr.Exists() && fr.String() != "" {
			finishReason = fr.String()
		}
	}
	if prose.String() != "Hello world" {
		t.Fatalf("prose = %q", prose.String())
	}
	if args.String() != `{"city":"NYC"}` {
		t.Fatalf("args = %q", args.String())
	}
	if finishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q want tool_calls", finishReason)
	}
}

func TestChatStreamEmitter_ProseOnly_FinishStop(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewChatStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1"}, emit)
	ev := em.Events()
	ev.OnProseDelta("only text")
	ev.OnComplete()

	finishReason := ""
	for _, f := range frames {
		if bytes.Equal(f, []byte("data: [DONE]\n\n")) {
			continue
		}
		if fr := gjson.GetBytes(jsonPart(f), "choices.0.finish_reason"); fr.Exists() && fr.String() != "" {
			finishReason = fr.String()
		}
	}
	if finishReason != "stop" {
		t.Fatalf("finish_reason = %q want stop", finishReason)
	}
}

func TestChatStreamEmitter_ToolHeaderHasEmptyArgs(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewChatStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1"}, emit)
	ev := em.Events()
	ev.OnToolCallStart(0, "call_abc", "f")
	ev.OnComplete()

	foundHeader := false
	for _, f := range frames {
		if bytes.Equal(f, []byte("data: [DONE]\n\n")) {
			continue
		}
		payload := jsonPart(f)
		if name := gjson.GetBytes(payload, "choices.0.delta.tool_calls.0.function.name"); name.Exists() && name.String() == "f" {
			args := gjson.GetBytes(payload, "choices.0.delta.tool_calls.0.function.arguments")
			if args.Exists() && args.String() == "" {
				foundHeader = true
			}
		}
	}
	if !foundHeader {
		t.Fatal("missing tool header frame with empty arguments")
	}
}

func TestChatStreamEmitter_CompleteIncludesUsageBeforeDone(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewChatStreamEmitter(UpstreamMeta{
		ResponseID:   "r1",
		Model:        "m1",
		UsagePayload: []byte(`{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":7}}`),
	}, emit)
	ev := em.Events()
	ev.OnProseDelta("done")
	ev.OnComplete()

	if len(frames) < 2 {
		t.Fatalf("frames = %d", len(frames))
	}
	usageFrame := frames[len(frames)-2]
	if bytes.Equal(usageFrame, []byte("data: [DONE]\n\n")) {
		t.Fatal("missing usage frame before DONE")
	}
	payload := jsonPart(usageFrame)
	if got := gjson.GetBytes(payload, "usage.prompt_tokens_details.cached_tokens").Int(); got != 7 {
		t.Fatalf("cached_tokens = %d", got)
	}
	if choices := gjson.GetBytes(payload, "choices"); !choices.Exists() || len(choices.Array()) != 0 {
		t.Fatalf("usage frame choices = %s", choices.Raw)
	}
}

func TestChatStreamEmitter_ReasoningDelta(t *testing.T) {
	var frames [][]byte
	emit := func(frame []byte) { frames = append(frames, bytes.Clone(frame)) }
	em := NewChatStreamEmitter(UpstreamMeta{ResponseID: "r1", Model: "m1"}, emit)
	ev := em.Events()
	ev.OnReasoningDelta("think")
	ev.OnComplete()

	found := false
	for _, frame := range frames {
		if bytes.Equal(frame, []byte("data: [DONE]\n\n")) {
			continue
		}
		if got := gjson.GetBytes(jsonPart(frame), "choices.0.delta.reasoning_content"); got.Exists() && got.String() == "think" {
			found = true
		}
	}
	if !found {
		t.Fatal("missing reasoning_content delta")
	}
}

func TestChatStreamEmitter_ToolChoiceNoneRejectsToolCall(t *testing.T) {
	var frames [][]byte
	emitter := NewChatStreamEmitter(UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r", Created: 1}, func(frame []byte) {
		frames = append(frames, append([]byte(nil), frame...))
	})
	emitter.SetToolChoice(ToolChoiceNone)
	events := emitter.Events()
	events.OnToolCallStart(0, "call_1", "f")
	events.OnArgsDelta(0, `{}`)
	events.OnToolCallEnd(0)

	joined := string(bytes.Join(frames, nil))
	if strings.Contains(joined, `"tool_calls"`) {
		t.Fatalf("tool call frames must not be emitted after none violation:\n%s", joined)
	}
	if !strings.Contains(joined, "tool_choice none forbids tool calls") {
		t.Fatalf("missing validation error frame:\n%s", joined)
	}
}
