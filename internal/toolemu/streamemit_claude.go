package toolemu

import (
	"encoding/json"
	"fmt"
)

// ClaudeStreamEmitter converts StreamParser callbacks into Anthropic Messages
// SSE frames (message_start → content_block_start/delta/stop → message_delta →
// message_stop) and pushes them to a frame sink.
type ClaudeStreamEmitter struct {
	meta     *UpstreamMeta
	emit     func(frame []byte)
	choice   ToolChoice
	failed   bool
	started  bool
	blockIdx int
	// Track which content block (if any) is currently open so prose and tool
	// calls can interleave correctly. openType is "" when no block is open.
	openType string // "text" | "tool_use" | ""
	hasTool  bool
}

// NewClaudeStreamEmitter creates an emitter that writes frames via emit.
func NewClaudeStreamEmitter(meta UpstreamMeta, emit func(frame []byte)) *ClaudeStreamEmitter {
	return &ClaudeStreamEmitter{meta: &meta, emit: emit, choice: ToolChoiceAuto}
}

// SetToolChoice configures native tool_choice validation for streamed emulation.
func (e *ClaudeStreamEmitter) SetToolChoice(choice ToolChoice) {
	e.choice = choice
}

// SetMeta updates stream metadata learned from upstream frames.
func (e *ClaudeStreamEmitter) SetMeta(meta UpstreamMeta) {
	e.meta = &meta
}

// Events returns the StreamEvents wired to this emitter.
func (e *ClaudeStreamEmitter) Events() StreamEvents {
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

func (e *ClaudeStreamEmitter) ensureStart() {
	if e.started {
		return
	}
	e.started = true
	message := map[string]any{
		"id":            e.meta.ResponseID,
		"type":          "message",
		"role":          "assistant",
		"model":         e.meta.Model,
		"content":       []any{},
		"stop_reason":   nil,
		"stop_sequence": nil,
	}
	if len(e.meta.UsagePayload) > 0 {
		var u any
		if err := json.Unmarshal(e.meta.UsagePayload, &u); err == nil {
			message["usage"] = u
		}
	}
	e.emit(e.frame("message_start", map[string]any{"type": "message_start", "message": message}))
}

func (e *ClaudeStreamEmitter) closeOpenBlock() {
	if e.openType == "" {
		return
	}
	e.emit(e.frame("content_block_stop", map[string]any{"type": "content_block_stop", "index": e.blockIdx}))
	e.blockIdx++
	e.openType = ""
}

func (e *ClaudeStreamEmitter) onProseDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureStart()
	if e.openType != "text" {
		e.closeOpenBlock()
		e.openType = "text"
		e.emit(e.frame("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         e.blockIdx,
			"content_block": map[string]any{"type": "text", "text": ""},
		}))
	}
	e.emit(e.frame("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": e.blockIdx,
		"delta": map[string]any{"type": "text_delta", "text": delta},
	}))
}

func (e *ClaudeStreamEmitter) onReasoningDelta(delta string) {
	if delta == "" || e.failed {
		return
	}
	e.ensureStart()
	if e.openType != "thinking" {
		e.closeOpenBlock()
		e.openType = "thinking"
		e.emit(e.frame("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         e.blockIdx,
			"content_block": map[string]any{"type": "thinking", "thinking": ""},
		}))
	}
	e.emit(e.frame("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": e.blockIdx,
		"delta": map[string]any{"type": "thinking_delta", "thinking": delta},
	}))
}

func (e *ClaudeStreamEmitter) onToolCallStart(index int, id, name string) {
	if e.failed {
		return
	}
	if e.choice.Kind == ToolChoiceKindNone || e.choice.Kind == ToolChoiceKindNamed && name != e.choice.Name {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureStart()
	e.closeOpenBlock()
	e.hasTool = true
	e.openType = "tool_use"
	// The stream parser derives ids via DeriveID (call_ prefix); Claude wants
	// toolu_ ids, so always derive a Claude-native id for emitted tool_use blocks.
	toolID := DeriveClaudeID(*e.meta, index)
	e.emit(e.frame("content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": e.blockIdx,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    toolID,
			"name":  name,
			"input": map[string]any{},
		},
	}))
}

func (e *ClaudeStreamEmitter) onArgsDelta(index int, delta string) {
	if delta == "" || e.failed {
		return
	}
	e.emit(e.frame("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": e.blockIdx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": delta},
	}))
}

func (e *ClaudeStreamEmitter) onToolCallEnd(index int) {
	if e.failed {
		return
	}
	if e.choice.Kind == ToolChoiceKindNamed && !e.hasTool {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.closeOpenBlock()
}

func (e *ClaudeStreamEmitter) onComplete() {
	if e.failed {
		return
	}
	if (e.choice.Kind == ToolChoiceKindRequired || e.choice.Kind == ToolChoiceKindNamed) && !e.hasTool {
		e.emitValidationError(ToolChoiceValidationError{Choice: e.choice})
		return
	}
	e.ensureStart()
	e.closeOpenBlock()
	stopReason := "end_turn"
	if e.hasTool {
		// Emulated tool calls always end in tool_use; the upstream stop_reason
		// (carried in FinishOverride) reflects the folded text completion and
		// must not override the emulated finish.
		stopReason = "tool_use"
	} else if e.meta.FinishOverride != "" {
		stopReason = e.meta.FinishOverride
	}
	body := map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
	}
	if len(e.meta.UsagePayload) > 0 {
		var u any
		if err := json.Unmarshal(e.meta.UsagePayload, &u); err == nil {
			body["usage"] = u
		}
	}
	e.emit(e.frame("message_delta", body))
	e.emit(e.frame("message_stop", map[string]any{"type": "message_stop"}))
}

func (e *ClaudeStreamEmitter) onError(_ error) {
	e.onComplete()
}

func (e *ClaudeStreamEmitter) emitValidationError(err error) {
	if e.failed {
		return
	}
	e.failed = true
	e.ensureStart()
	e.closeOpenBlock()
	body := map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "invalid_request_error",
			"message": err.Error(),
		},
	}
	e.emit(e.frame("error", body))
	e.emit(e.frame("message_stop", map[string]any{"type": "message_stop"}))
}

func (e *ClaudeStreamEmitter) frame(eventType string, body map[string]any) []byte {
	raw, _ := json.Marshal(body)
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, raw))
}
