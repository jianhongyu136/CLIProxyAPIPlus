package toolemu

import (
	"encoding/json"
	"fmt"
)

// ChatStreamEmitter converts StreamParser callbacks into OpenAI chat-completion
// SSE frames and pushes them to a frame sink.
type ChatStreamEmitter struct {
	meta        *UpstreamMeta
	emit        func(frame []byte)
	roleSent    bool
	hasToolCall bool
	choice      ToolChoice
	failed      bool
	toolNames   map[int]string
}

// NewChatStreamEmitter creates an emitter that writes frames via emit.
func NewChatStreamEmitter(meta UpstreamMeta, emit func(frame []byte)) *ChatStreamEmitter {
	return &ChatStreamEmitter{meta: &meta, emit: emit, choice: ToolChoiceAuto, toolNames: map[int]string{}}
}

// SetToolChoice configures native tool_choice validation for streamed emulation.
func (e *ChatStreamEmitter) SetToolChoice(choice ToolChoice) {
	e.choice = choice
}

// SetMeta updates stream metadata learned from upstream frames.
func (e *ChatStreamEmitter) SetMeta(meta UpstreamMeta) {
	e.meta = &meta
}

// Events returns the StreamEvents wired to this emitter.
func (e *ChatStreamEmitter) Events() StreamEvents {
	return StreamEvents{
		OnMetaUpdate:     e.SetMeta,
		OnProseDelta:     e.onProseDelta,
		OnReasoningDelta: e.onReasoningDelta,
		OnToolCallStart:  e.onToolCallStart,
		OnArgsDelta:      e.onArgsDelta,
		OnToolCallEnd:    e.onToolCallEnd,
		OnComplete:       e.onComplete,
		OnError:          e.onError,
	}
}

func (e *ChatStreamEmitter) ensureRole() {
	if e.roleSent {
		return
	}
	e.roleSent = true
	e.emit(e.frame(map[string]any{"role": "assistant"}, ""))
}

func (e *ChatStreamEmitter) onProseDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureRole()
	e.emit(e.frame(map[string]any{"content": delta}, ""))
}

func (e *ChatStreamEmitter) onReasoningDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureRole()
	e.emit(e.frame(map[string]any{"reasoning_content": delta}, ""))
}

func (e *ChatStreamEmitter) onToolCallStart(index int, id, name string) {
	if e.failed {
		return
	}
	if e.choice.Kind == ToolChoiceKindNone || e.choice.Kind == ToolChoiceKindNamed && name != e.choice.Name {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureRole()
	e.hasToolCall = true
	e.toolNames[index] = name
	header := []any{map[string]any{
		"index": index, "id": id, "type": "function",
		"function": map[string]any{"name": name, "arguments": ""},
	}}
	e.emit(e.frame(map[string]any{"tool_calls": header}, ""))
}

func (e *ChatStreamEmitter) onArgsDelta(index int, delta string) {
	if delta == "" || e.failed {
		return
	}
	tc := []any{map[string]any{
		"index":    index,
		"function": map[string]any{"arguments": delta},
	}}
	e.emit(e.frame(map[string]any{"tool_calls": tc}, ""))
}

func (e *ChatStreamEmitter) onToolCallEnd(index int) {
	if e.failed {
		return
	}
	if e.choice.Kind == ToolChoiceKindNamed && e.toolNames[index] != e.choice.Name {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
	}
}

func (e *ChatStreamEmitter) onComplete() {
	if e.failed {
		return
	}
	if (e.choice.Kind == ToolChoiceKindRequired || e.choice.Kind == ToolChoiceKindNamed) && !e.hasToolCall {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureRole()
	finish := "stop"
	if e.hasToolCall {
		finish = "tool_calls"
	}
	e.emit(e.frame(map[string]any{}, finish))
	if len(e.meta.UsagePayload) > 0 {
		var usage any
		if err := json.Unmarshal(e.meta.UsagePayload, &usage); err == nil {
			e.emit(e.usageFrame(usage))
		}
	}
	e.emit([]byte("data: [DONE]\n\n"))
}

func (e *ChatStreamEmitter) onError(_ error) {
	e.onComplete()
}

func (e *ChatStreamEmitter) emitValidationError(err error) {
	if e.failed {
		return
	}
	e.failed = true
	body := map[string]any{"error": map[string]any{"message": err.Error(), "type": "invalid_request_error", "code": "tool_choice_violation"}}
	raw, _ := json.Marshal(body)
	e.emit([]byte(fmt.Sprintf("data: %s\n\n", raw)))
	e.emit([]byte("data: [DONE]\n\n"))
}

func (e *ChatStreamEmitter) frame(delta map[string]any, finish string) []byte {
	choice := map[string]any{"index": 0, "delta": delta}
	if finish != "" {
		choice["finish_reason"] = finish
	} else {
		choice["finish_reason"] = nil
	}
	body := map[string]any{
		"id":      e.meta.ResponseID,
		"object":  "chat.completion.chunk",
		"created": e.meta.Created,
		"model":   e.meta.Model,
		"choices": []any{choice},
	}
	raw, _ := json.Marshal(body)
	return []byte(fmt.Sprintf("data: %s\n\n", raw))
}

func (e *ChatStreamEmitter) usageFrame(usage any) []byte {
	body := map[string]any{
		"id":      e.meta.ResponseID,
		"object":  "chat.completion.chunk",
		"created": e.meta.Created,
		"model":   e.meta.Model,
		"choices": []any{},
		"usage":   usage,
	}
	raw, _ := json.Marshal(body)
	return []byte(fmt.Sprintf("data: %s\n\n", raw))
}
