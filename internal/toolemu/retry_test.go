package toolemu

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"
)

func newChatBody(text string) []byte {
	return []byte(`{"id":"r","model":"m","choices":[{"message":{"role":"assistant","content":` + jsonString(text) + `}}]}`)
}

func newResponsesBody(text string) []byte {
	return []byte(`{"id":"r","model":"m","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":` + jsonString(text) + `}]}]}`)
}

// jsonString returns a JSON-encoded string literal (including surrounding quotes)
// for embedding within hand-built JSON test fixtures.
func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// malformedToolCall returns text containing a tool_call block with invalid
// JSON inside; this triggers ParseAndRetry's retry branch.
const malformedToolCall = `<tool_call>not json</tool_call>`

func TestParseAndRetry_ChatFirstAttemptSuccess(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	calls := 0
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		calls++
		return newChatBody(`<tool_call>{"name":"f","arguments":{}}</tool_call>`), nil
	}
	res, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 1}, ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("sender invoked %d times, want 1", calls)
	}
	if len(res.Parsed.ToolCalls) != 1 || res.Parsed.ToolCalls[0].Name != "f" {
		t.Fatalf("parsed = %+v", res.Parsed)
	}
	if res.Degraded {
		t.Fatal("must not be degraded on first-attempt success")
	}
}

func TestParseAndRetry_RetryThenSuccessAppendsInstruction(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	var captured [][]byte
	send := func(_ context.Context, p []byte) ([]byte, error) {
		captured = append(captured, append([]byte(nil), p...))
		if len(captured) == 1 {
			return newChatBody(malformedToolCall), nil
		}
		return newChatBody(`<tool_call>{"name":"f","arguments":{}}</tool_call>`), nil
	}
	res, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 1}, ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if len(captured) != 2 {
		t.Fatalf("want 2 sends, got %d", len(captured))
	}
	if !bytes.Contains(captured[1], []byte("Your previous response did not contain a valid <tool_call> block")) {
		t.Fatalf("retry payload missing instruction:\n%s", captured[1])
	}
	if len(res.Parsed.ToolCalls) != 1 {
		t.Fatalf("expected one ToolCall after retry, got %+v", res.Parsed)
	}
}

func TestParseAndRetry_DegradeOnPersistentFailure(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		return newChatBody(malformedToolCall), nil
	}
	res, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 1, OnFailure: "parse_failed_to_content"}, ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Degraded {
		t.Fatal("expected Degraded=true")
	}
	if res.Parsed.Prose != malformedToolCall {
		t.Fatalf("prose = %q", res.Parsed.Prose)
	}
}

func TestParseAndRetry_ErrorOnPersistentFailure(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		return newChatBody(malformedToolCall), nil
	}
	_, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 1, OnFailure: "error"}, ToolChoiceAuto)
	if err == nil {
		t.Fatal("expected ParseFailedError")
	}
	var pfe ParseFailedError
	if !errors.As(err, &pfe) {
		t.Fatalf("expected ParseFailedError, got %T: %v", err, err)
	}
	if pfe.LastText != malformedToolCall {
		t.Fatalf("LastText = %q", pfe.LastText)
	}
}

func TestParseAndRetry_ResponsesShape(t *testing.T) {
	payload := []byte(`{"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`)
	var captured [][]byte
	send := func(_ context.Context, p []byte) ([]byte, error) {
		captured = append(captured, append([]byte(nil), p...))
		if len(captured) == 1 {
			return newResponsesBody(malformedToolCall), nil
		}
		return newResponsesBody(`<tool_call>{"name":"g","arguments":{}}</tool_call>`), nil
	}
	res, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIResponses, RetryPolicy{Attempts: 1}, ToolChoiceAuto)
	if err != nil {
		t.Fatal(err)
	}
	if len(captured) != 2 {
		t.Fatalf("want 2 sends, got %d", len(captured))
	}
	if !bytes.Contains(captured[1], []byte("Your previous response did not contain a valid <tool_call> block")) {
		t.Fatalf("retry payload missing instruction:\n%s", captured[1])
	}
	if len(res.Parsed.ToolCalls) != 1 || res.Parsed.ToolCalls[0].Name != "g" {
		t.Fatalf("parsed = %+v", res.Parsed)
	}
}

func TestParseAndRetry_ToolChoiceRequiredRetriesThenDegrades(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	calls := 0
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		calls++
		return newChatBody("plain answer"), nil
	}

	res, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 1, OnFailure: "parse_failed_to_content"}, ToolChoiceRequired)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("send calls = %d, want 2", calls)
	}
	if !res.Degraded || res.Parsed.Prose != "plain answer" {
		t.Fatalf("unexpected result: %+v", res)
	}
}

func TestParseAndRetry_ToolChoiceNamedRejectsWrongTool(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		return newChatBody(`<tool_call>{"name":"other","arguments":{}}</tool_call>`), nil
	}

	_, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 0, OnFailure: "error"}, ToolChoiceNamed("expected"))
	if err == nil {
		t.Fatal("expected ParseFailedError")
	}
	var pfe ParseFailedError
	if !errors.As(err, &pfe) {
		t.Fatalf("expected ParseFailedError, got %T: %v", err, err)
	}
}

func TestParseAndRetry_ToolChoiceNoneRejectsToolCall(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}]}`)
	send := func(_ context.Context, _ []byte) ([]byte, error) {
		return newChatBody(`<tool_call>{"name":"f","arguments":{}}</tool_call>`), nil
	}

	_, err := ParseAndRetry(context.Background(), payload, send, ShapeOpenAIChat, RetryPolicy{Attempts: 0, OnFailure: "error"}, ToolChoiceNone)
	if err == nil {
		t.Fatal("expected ParseFailedError")
	}
}
