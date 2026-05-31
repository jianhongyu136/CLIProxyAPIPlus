package toolemu

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/tidwall/gjson"
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
	meta := p.Parser.meta
	scanner := bufio.NewScanner(p.Reader)
	scanner.Buffer(nil, 52_428_800) // 50 MB
	var eventType string

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			p.Parser.Close()
			return meta, ctx.Err()
		default:
		}

		line := scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			if line == "" {
				eventType = ""
			}
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		done := false
		switch p.Shape {
		case ShapeOpenAIChat:
			done = p.processChatDelta(data, &meta)
		case ShapeOpenAIResponses:
			done = p.processResponsesDelta(data, eventType, &meta)
		}
		if done {
			break
		}
	}

	p.Parser.UpdateMeta(meta)
	p.Parser.Close()
	return meta, scanner.Err()
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
