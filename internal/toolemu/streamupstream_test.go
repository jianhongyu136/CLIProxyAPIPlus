package toolemu

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestUpstreamPump_ChatShape(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"r1","model":"m","choices":[{"delta":{"role":"assistant"},"index":0}]}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{"content":"Hello"},"index":0}]}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{"content":" world"},"index":0}]}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{
		Reader: bytes.NewReader([]byte(sse)),
		Shape:  ShapeOpenAIChat,
		Parser: parser,
	}
	meta, err := pump.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if spy.allProse() != "Hello world" {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if meta.ResponseID != "r1" {
		t.Fatalf("got response id %q", meta.ResponseID)
	}
	if meta.Model != "m" {
		t.Fatalf("got model %q", meta.Model)
	}
}

func TestUpstreamPump_ResponsesShape(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"r2","model":"m2"}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"Hi "}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"there"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r2","model":"m2","usage":{"input_tokens":8}}}`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{
		Reader: bytes.NewReader([]byte(sse)),
		Shape:  ShapeOpenAIResponses,
		Parser: parser,
	}
	meta, err := pump.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if spy.allProse() != "Hi there" {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if meta.ResponseID != "r2" {
		t.Fatalf("got response id %q", meta.ResponseID)
	}
}

func TestUpstreamPump_ChatReasoningDelta(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"r1","model":"m","choices":[{"delta":{"reasoning_content":"think"},"index":0}]}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{"content":"answer"},"index":0}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{Reader: bytes.NewReader([]byte(sse)), Shape: ShapeOpenAIChat, Parser: parser}
	if _, err := pump.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if spy.allReasoning() != "think" {
		t.Fatalf("got reasoning %q", spy.allReasoning())
	}
	if spy.allProse() != "answer" {
		t.Fatalf("got prose %q", spy.allProse())
	}
}

func TestUpstreamPump_ResponsesReasoningDelta(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.reasoning_summary_text.delta`,
		`data: {"type":"response.reasoning_summary_text.delta","delta":"think"}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"answer"}`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{Reader: bytes.NewReader([]byte(sse)), Shape: ShapeOpenAIResponses, Parser: parser}
	if _, err := pump.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if spy.allReasoning() != "think" {
		t.Fatalf("got reasoning %q", spy.allReasoning())
	}
	if spy.allProse() != "answer" {
		t.Fatalf("got prose %q", spy.allProse())
	}
}

func TestUpstreamPump_ResponsesCompletedStopsReading(t *testing.T) {
	sse := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":"before"}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"r2","model":"m2","usage":{"input_tokens":8}}}`,
		``,
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","delta":" after"}`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{Reader: bytes.NewReader([]byte(sse)), Shape: ShapeOpenAIResponses, Parser: parser}
	meta, err := pump.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := spy.allProse(); got != "before" {
		t.Fatalf("got prose %q", got)
	}
	if meta.ResponseID != "r2" {
		t.Fatalf("got response id %q", meta.ResponseID)
	}
	if len(meta.UsagePayload) == 0 {
		t.Fatal("missing usage from completed event")
	}
}

func TestUpstreamPump_ChatFinishReasonStopsReading(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"r1","model":"m","choices":[{"delta":{"content":"before"},"index":0}]}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`,
		``,
		`data: {"id":"r1","model":"m","choices":[{"delta":{"content":" after"},"index":0}]}`,
		``,
	}, "\n")

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{Reader: bytes.NewReader([]byte(sse)), Shape: ShapeOpenAIChat, Parser: parser}
	meta, err := pump.Run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if got := spy.allProse(); got != "before" {
		t.Fatalf("got prose %q", got)
	}
	if meta.ResponseID != "r1" {
		t.Fatalf("got response id %q", meta.ResponseID)
	}
	if len(meta.UsagePayload) == 0 {
		t.Fatal("missing usage from finish event")
	}
}

func TestUpstreamPump_ContextCancellation(t *testing.T) {
	line := "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n"
	sse := strings.Repeat(line, 1000)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	spy := &spyEvents{}
	parser := NewStreamParser(spy.events(), UpstreamMeta{})
	pump := &UpstreamPump{
		Reader: bytes.NewReader([]byte(sse)),
		Shape:  ShapeOpenAIChat,
		Parser: parser,
	}
	_, err := pump.Run(ctx)
	if err == nil {
		t.Fatal("expected context error")
	}
}
