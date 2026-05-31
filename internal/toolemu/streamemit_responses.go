package toolemu

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesStreamEmitter converts StreamParser callbacks into OpenAI Responses
// SSE event frames.
type ResponsesStreamEmitter struct {
	meta *UpstreamMeta
	emit func(frame []byte)

	choice      ToolChoice
	failed      bool
	hasToolCall bool

	startEmitted bool
	outputIndex  int

	inMessage    bool
	messageProse strings.Builder

	inToolCall bool
	toolCallID string
	toolName   string
	toolArgs   strings.Builder

	inReasoning   bool
	reasoningID   string
	reasoningText strings.Builder
}

// NewResponsesStreamEmitter creates an emitter for OpenAI Responses SSE.
func NewResponsesStreamEmitter(meta UpstreamMeta, emit func(frame []byte)) *ResponsesStreamEmitter {
	return &ResponsesStreamEmitter{meta: &meta, emit: emit, choice: ToolChoiceAuto}
}

// SetToolChoice configures native tool_choice validation for streamed emulation.
func (e *ResponsesStreamEmitter) SetToolChoice(choice ToolChoice) {
	e.choice = choice
}

// SetMeta updates stream metadata learned from upstream frames.
func (e *ResponsesStreamEmitter) SetMeta(meta UpstreamMeta) {
	e.meta = &meta
}

// Events returns the StreamEvents wired to this emitter.
func (e *ResponsesStreamEmitter) Events() StreamEvents {
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

func (e *ResponsesStreamEmitter) ensureStart() {
	if e.startEmitted {
		return
	}
	e.startEmitted = true
	stub := e.responseStub("in_progress")
	e.emit(e.sse("response.created", map[string]any{"type": "response.created", "response": stub}))
	e.emit(e.sse("response.in_progress", map[string]any{"type": "response.in_progress", "response": stub}))
}

func (e *ResponsesStreamEmitter) responseStub(status string) map[string]any {
	return map[string]any{
		"id": e.meta.ResponseID, "object": "response",
		"created_at": e.meta.Created, "model": e.meta.Model,
		"status": status,
	}
}

func (e *ResponsesStreamEmitter) ensureMessage() {
	if e.failed || e.inMessage {
		return
	}
	e.ensureStart()
	if e.inToolCall {
		e.closeToolCall()
	}
	if e.failed {
		return
	}
	if e.inReasoning {
		e.closeReasoning()
	}
	e.inMessage = true
	itemID := "msg_" + e.meta.ResponseID
	msgItem := map[string]any{
		"type": "message", "role": "assistant", "id": itemID, "status": "in_progress",
		"content": []any{},
	}
	e.emit(e.sse("response.output_item.added", map[string]any{
		"type": "response.output_item.added", "output_index": e.outputIndex, "item": msgItem,
	}))
	e.emit(e.sse("response.content_part.added", map[string]any{
		"type": "response.content_part.added", "output_index": e.outputIndex, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": ""},
	}))
}

func (e *ResponsesStreamEmitter) closeMessage() {
	if !e.inMessage {
		return
	}
	itemID := "msg_" + e.meta.ResponseID
	text := e.messageProse.String()
	e.emit(e.sse("response.output_text.done", map[string]any{
		"type": "response.output_text.done", "output_index": e.outputIndex, "content_index": 0, "text": text,
	}))
	e.emit(e.sse("response.content_part.done", map[string]any{
		"type": "response.content_part.done", "output_index": e.outputIndex, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": text},
	}))
	e.emit(e.sse("response.output_item.done", map[string]any{
		"type": "response.output_item.done", "output_index": e.outputIndex,
		"item": map[string]any{
			"type": "message", "role": "assistant", "id": itemID, "status": "completed",
			"content": []any{map[string]any{"type": "output_text", "text": text}},
		},
	}))
	e.inMessage = false
	e.messageProse.Reset()
	e.outputIndex++
}

func (e *ResponsesStreamEmitter) onProseDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureMessage()
	if e.failed {
		return
	}
	e.messageProse.WriteString(delta)
	e.emit(e.sse("response.output_text.delta", map[string]any{
		"type": "response.output_text.delta", "output_index": e.outputIndex, "content_index": 0,
		"delta": delta,
	}))
}

func (e *ResponsesStreamEmitter) onReasoningDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureStart()
	if e.inMessage {
		e.closeMessage()
	}
	if e.inToolCall {
		e.closeToolCall()
	}
	if e.failed {
		return
	}
	if !e.inReasoning {
		e.inReasoning = true
		e.reasoningID = "rs_" + e.meta.ResponseID
		e.reasoningText.Reset()
		e.emit(e.sse("response.output_item.added", map[string]any{
			"type": "response.output_item.added", "output_index": e.outputIndex,
			"item": map[string]any{"id": e.reasoningID, "type": "reasoning", "status": "in_progress", "summary": []any{}},
		}))
		e.emit(e.sse("response.reasoning_summary_part.added", map[string]any{
			"type": "response.reasoning_summary_part.added", "item_id": e.reasoningID, "output_index": e.outputIndex, "summary_index": 0,
			"part": map[string]any{"type": "summary_text", "text": ""},
		}))
	}
	e.reasoningText.WriteString(delta)
	e.emit(e.sse("response.reasoning_summary_text.delta", map[string]any{
		"type": "response.reasoning_summary_text.delta", "item_id": e.reasoningID, "output_index": e.outputIndex, "summary_index": 0,
		"delta": delta,
	}))
}

func (e *ResponsesStreamEmitter) onToolCallStart(index int, id, name string) {
	if e.failed {
		return
	}
	if e.choice.Kind == ToolChoiceKindNone || e.choice.Kind == ToolChoiceKindNamed && name != e.choice.Name {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureStart()
	if e.inMessage {
		e.closeMessage()
	}
	if e.inReasoning {
		e.closeReasoning()
	}
	if e.failed {
		return
	}
	e.inToolCall = true
	e.hasToolCall = true
	e.toolCallID = id
	e.toolName = name
	e.toolArgs.Reset()
	callItem := map[string]any{
		"type": "function_call", "id": id, "call_id": id,
		"name": name, "arguments": "", "status": "in_progress",
	}
	e.emit(e.sse("response.output_item.added", map[string]any{
		"type": "response.output_item.added", "output_index": e.outputIndex, "item": callItem,
	}))
}

func (e *ResponsesStreamEmitter) onArgsDelta(index int, delta string) {
	if delta == "" || !e.inToolCall || e.failed {
		return
	}
	e.toolArgs.WriteString(delta)
	e.emit(e.sse("response.function_call_arguments.delta", map[string]any{
		"type":         "response.function_call_arguments.delta",
		"output_index": e.outputIndex, "item_id": e.toolCallID, "delta": delta,
	}))
}

func (e *ResponsesStreamEmitter) onToolCallEnd(_ int) {
	e.closeToolCall()
}

func (e *ResponsesStreamEmitter) closeToolCall() {
	if !e.inToolCall || e.failed {
		return
	}
	args := e.toolArgs.String()
	if e.choice.Kind == ToolChoiceKindNamed && e.toolName != e.choice.Name {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.emit(e.sse("response.function_call_arguments.done", map[string]any{
		"type":         "response.function_call_arguments.done",
		"output_index": e.outputIndex, "item_id": e.toolCallID, "arguments": args,
	}))
	e.emit(e.sse("response.output_item.done", map[string]any{
		"type": "response.output_item.done", "output_index": e.outputIndex,
		"item": map[string]any{
			"type": "function_call", "id": e.toolCallID, "call_id": e.toolCallID,
			"name": e.toolName, "arguments": args, "status": "completed",
		},
	}))
	e.inToolCall = false
	e.toolCallID = ""
	e.toolName = ""
	e.toolArgs.Reset()
	e.outputIndex++
}

func (e *ResponsesStreamEmitter) closeReasoning() {
	if !e.inReasoning {
		return
	}
	text := e.reasoningText.String()
	e.emit(e.sse("response.reasoning_summary_text.done", map[string]any{
		"type": "response.reasoning_summary_text.done", "item_id": e.reasoningID, "output_index": e.outputIndex, "summary_index": 0,
		"text": text,
	}))
	e.emit(e.sse("response.reasoning_summary_part.done", map[string]any{
		"type": "response.reasoning_summary_part.done", "item_id": e.reasoningID, "output_index": e.outputIndex, "summary_index": 0,
		"part": map[string]any{"type": "summary_text", "text": text},
	}))
	e.emit(e.sse("response.output_item.done", map[string]any{
		"type": "response.output_item.done", "output_index": e.outputIndex,
		"item": map[string]any{
			"id": e.reasoningID, "type": "reasoning", "status": "completed",
			"summary": []any{map[string]any{"type": "summary_text", "text": text}},
		},
	}))
	e.inReasoning = false
	e.reasoningID = ""
	e.reasoningText.Reset()
	e.outputIndex++
}

func (e *ResponsesStreamEmitter) onComplete() {
	if e.failed {
		return
	}
	if (e.choice.Kind == ToolChoiceKindRequired || e.choice.Kind == ToolChoiceKindNamed) && !e.hasToolCall {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureStart()
	if e.inMessage {
		e.closeMessage()
	}
	if e.inToolCall {
		e.closeToolCall()
	}
	if e.failed {
		return
	}
	if e.inReasoning {
		e.closeReasoning()
	}
	completed := e.responseStub("completed")
	if len(e.meta.UsagePayload) > 0 {
		var u any
		if err := json.Unmarshal(e.meta.UsagePayload, &u); err == nil {
			completed["usage"] = u
		}
	}
	e.emit(e.sse("response.completed", map[string]any{
		"type": "response.completed", "response": completed,
	}))
}

func (e *ResponsesStreamEmitter) onError(_ error) {
	e.onComplete()
}

func (e *ResponsesStreamEmitter) emitValidationError(err error) {
	if e.failed {
		return
	}
	e.failed = true
	e.ensureStart()
	e.emit(e.sse("error", map[string]any{
		"type": "error",
		"error": map[string]any{
			"message": err.Error(),
			"type":    "invalid_request_error",
			"code":    "tool_choice_violation",
		},
	}))
}

func (e *ResponsesStreamEmitter) sse(eventType string, body map[string]any) []byte {
	raw, _ := json.Marshal(body)
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, raw))
}
