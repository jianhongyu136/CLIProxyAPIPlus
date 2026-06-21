package toolemu

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// UpstreamPump reads SSE events from an upstream streaming response, extracts
// assistant text deltas according to the upstream Shape, and feeds them into a
// StreamParser. It returns the accumulated UpstreamMeta when the stream ends.
type UpstreamPump struct {
	Reader io.Reader
	Shape  UpstreamShape
	Parser *StreamParser
}

// Run processes the upstream SSE stream until completion or context cancellation.
func (p *UpstreamPump) Run(ctx context.Context) (UpstreamMeta, error) {
	if p == nil || p.Parser == nil {
		return UpstreamMeta{}, fmt.Errorf("toolemu stream: nil parser")
	}
	meta := p.Parser.meta
	scanner := bufio.NewScanner(p.Reader)
	scanner.Buffer(nil, 52_428_800) // 50 MB
	var eventType string
	var dataLines []string

	dispatch := func() (bool, error) {
		if len(dataLines) == 0 {
			eventType = ""
			return false, nil
		}
		data := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		defer func() { eventType = "" }()
		if data == "[DONE]" {
			return true, nil
		}
		if errEvent := upstreamSSEError(data, eventType); errEvent != nil {
			p.Parser.Close()
			return false, errEvent
		}
		done := false
		switch p.Shape {
		case ShapeOpenAIChat:
			done = p.processChatDelta(data, &meta)
		case ShapeOpenAIResponses:
			done = p.processResponsesDelta(data, eventType, &meta)
		case ShapeClaudeMessages:
			done = p.processClaudeDelta(data, eventType, &meta)
		}
		return done, nil
	}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			p.Parser.Close()
			return meta, ctx.Err()
		default:
		}

		line := scanner.Text()
		if line == "" {
			done, errDispatch := dispatch()
			if errDispatch != nil {
				return meta, errDispatch
			}
			if done {
				break
			}
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		field, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		if strings.HasPrefix(value, " ") {
			value = value[1:]
		}
		switch field {
		case "event":
			eventType = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}

	if errScan := scanner.Err(); errScan != nil {
		p.Parser.Close()
		return meta, errScan
	}
	if len(dataLines) > 0 {
		done, errDispatch := dispatch()
		if errDispatch != nil {
			return meta, errDispatch
		}
		_ = done
	}

	p.Parser.UpdateMeta(meta)
	p.Parser.Close()
	return meta, nil
}

func upstreamSSEError(data, eventType string) error {
	root := gjson.Parse(data)
	if eventType != "error" && root.Get("type").String() != "error" {
		return nil
	}
	message := strings.TrimSpace(root.Get("error.message").String())
	if message == "" {
		message = strings.TrimSpace(root.Get("error.type").String())
	}
	if message == "" {
		message = "unknown upstream stream error"
	}
	return fmt.Errorf("toolemu stream: upstream error event: %s", message)
}

func (p *UpstreamPump) processChatDelta(data string, meta *UpstreamMeta) bool {
	root := gjson.Parse(data)

	if id := root.Get("id").String(); id != "" && meta.ResponseID == "" {
		meta.ResponseID = id
	}
	if model := root.Get("model").String(); model != "" && meta.Model == "" {
		meta.Model = model
	}
	if created := root.Get("created").Int(); created != 0 && meta.Created == 0 {
		meta.Created = created
	}
	if usage := root.Get("usage"); usage.Exists() && len(usage.Raw) > 2 {
		meta.UsagePayload = []byte(usage.Raw)
	}

	delta := root.Get("choices.0.delta")
	if reasoning := delta.Get("reasoning_content"); reasoning.Exists() {
		for _, text := range collectReasoningTexts(reasoning) {
			p.emitReasoning(*meta, text)
		}
	}
	content := delta.Get("content")
	if content.Exists() && content.Type == gjson.String {
		p.Parser.UpdateMeta(*meta)
		p.Parser.Feed(content.String())
	}

	finish := root.Get("choices.0.finish_reason")
	return finish.Exists() && finish.String() != ""
}

func (p *UpstreamPump) processResponsesDelta(data, eventType string, meta *UpstreamMeta) bool {
	root := gjson.Parse(data)
	evType := eventType
	if evType == "" {
		evType = root.Get("type").String()
	}

	switch evType {
	case "response.created", "response.completed":
		resp := root.Get("response")
		if id := resp.Get("id").String(); id != "" {
			meta.ResponseID = id
		}
		if model := resp.Get("model").String(); model != "" {
			meta.Model = model
		}
		if created := resp.Get("created_at").Int(); created != 0 {
			meta.Created = created
		}
		if usage := resp.Get("usage"); usage.Exists() && len(usage.Raw) > 2 {
			meta.UsagePayload = []byte(usage.Raw)
		}
		if evType == "response.completed" {
			return true
		}
	case "response.output_text.delta":
		delta := root.Get("delta")
		if delta.Exists() && delta.Type == gjson.String {
			p.Parser.UpdateMeta(*meta)
			p.Parser.Feed(delta.String())
		}
	case "response.reasoning_summary_text.delta":
		delta := root.Get("delta")
		if delta.Exists() && delta.Type == gjson.String {
			p.emitReasoning(*meta, delta.String())
		}
	}
	return false
}

func (p *UpstreamPump) processClaudeDelta(data, eventType string, meta *UpstreamMeta) bool {
	root := gjson.Parse(data)
	evType := eventType
	if evType == "" {
		evType = root.Get("type").String()
	}

	switch evType {
	case "message_start":
		msg := root.Get("message")
		if id := msg.Get("id").String(); id != "" {
			meta.ResponseID = id
		}
		if model := msg.Get("model").String(); model != "" {
			meta.Model = model
		}
		if usage := msg.Get("usage"); usage.Exists() && len(usage.Raw) > 2 {
			meta.UsagePayload = []byte(usage.Raw)
		}
	case "message_delta":
		if usage := root.Get("usage"); usage.Exists() && len(usage.Raw) > 2 {
			// Anthropic's message_delta usage typically carries only the
			// (cumulative) output_tokens; input/cache fields live on
			// message_start. Merge rather than overwrite so the final payload
			// retains input_tokens and cache counters for accurate accounting.
			meta.UsagePayload = mergeClaudeUsage(meta.UsagePayload, usage)
		}
		if sr := root.Get("delta.stop_reason").String(); sr != "" {
			meta.FinishOverride = claudeStopReasonToFinish(sr)
		}
	case "content_block_delta":
		delta := root.Get("delta")
		if !delta.Exists() {
			return false
		}
		switch delta.Get("type").String() {
		case "text_delta":
			if text := delta.Get("text"); text.Exists() && text.Type == gjson.String {
				p.Parser.UpdateMeta(*meta)
				p.Parser.Feed(text.String())
			}
		case "thinking_delta":
			if text := delta.Get("thinking"); text.Exists() && text.Type == gjson.String {
				p.emitReasoning(*meta, text.String())
			}
		}
	case "message_stop":
		return true
	}
	return false
}

// mergeClaudeUsage folds the fields of a message_delta usage node into the
// usage payload captured at message_start. Prior fields are preserved; any
// field present in the delta overwrites its prior value (Anthropic reports the
// cumulative output_tokens on message_delta). When there is no prior payload,
// the delta is returned as-is.
func mergeClaudeUsage(prior []byte, delta gjson.Result) []byte {
	if len(prior) == 0 {
		return []byte(delta.Raw)
	}
	merged := prior
	delta.ForEach(func(key, value gjson.Result) bool {
		if value.Type == gjson.Null {
			return true
		}
		if updated, err := sjson.SetRawBytes(merged, key.String(), []byte(value.Raw)); err == nil {
			merged = updated
		}
		return true
	})
	return merged
}

// claudeStopReasonToFinish maps an Anthropic stop_reason onto the toolemu
// FinishOverride vocabulary used by emitters. Tool-use responses keep their
// native tool_use finish; everything else is treated as a normal stop.
func claudeStopReasonToFinish(sr string) string {
	switch sr {
	case "tool_use", "end_turn", "stop_sequence", "max_tokens":
		return sr
	default:
		return sr
	}
}

func (p *UpstreamPump) emitReasoning(meta UpstreamMeta, delta string) {
	if delta == "" || p.Parser.events.OnReasoningDelta == nil {
		return
	}
	p.Parser.UpdateMeta(meta)
	p.Parser.events.OnReasoningDelta(delta)
}

func collectReasoningTexts(node gjson.Result) []string {
	var texts []string
	if !node.Exists() {
		return texts
	}
	if node.IsArray() {
		node.ForEach(func(_, value gjson.Result) bool {
			texts = append(texts, collectReasoningTexts(value)...)
			return true
		})
		return texts
	}
	switch node.Type {
	case gjson.String:
		if text := node.String(); text != "" {
			texts = append(texts, text)
		}
	case gjson.JSON:
		if text := node.Get("text"); text.Exists() && text.String() != "" {
			texts = append(texts, text.String())
		}
	}
	return texts
}

// UpstreamPumpError wraps errors from the pump with context.
type UpstreamPumpError struct {
	Cause error
	Phase string
}

func (e *UpstreamPumpError) Error() string {
	return fmt.Sprintf("toolemu upstream pump: %s: %v", e.Phase, e.Cause)
}

func (e *UpstreamPumpError) Unwrap() error { return e.Cause }
