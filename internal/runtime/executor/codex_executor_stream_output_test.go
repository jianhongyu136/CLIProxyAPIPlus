package executor

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	_ "github.com/router-for-me/CLIProxyAPI/v7/internal/translator"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestCodexExecutorExecute_EmptyStreamCompletionOutputUsesOutputItemDone(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]},\"output_index\":0}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1775555723,\"status\":\"completed\",\"model\":\"gpt-5.4-mini-2026-03-17\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":28,\"total_tokens\":36}}}\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	resp, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4-mini",
		Payload: []byte(`{"model":"gpt-5.4-mini","messages":[{"role":"user","content":"Say ok"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
		Stream:       false,
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	gotContent := gjson.GetBytes(resp.Payload, "choices.0.message.content").String()
	if gotContent != "ok" {
		t.Fatalf("choices.0.message.content = %q, want %q; payload=%s", gotContent, "ok", string(resp.Payload))
	}
}

func TestCodexExecutorExecuteSurfacesTerminalStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: response.created\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}` + "\n\n"))
		_, _ = w.Write([]byte("event: error\n"))
		_, _ = w.Write([]byte(`data: {"type":"error","error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","param":"input"},"sequence_number":2}` + "\n\n"))
		_, _ = w.Write([]byte("event: response.failed\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again."}}}` + "\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.5",
		Payload: []byte(`{"model":"gpt-5.5","input":"hello"}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
	})
	if err == nil {
		t.Fatal("expected terminal stream error, got nil")
	}
	if got := statusCodeFromTestError(t, err); got != http.StatusBadRequest {
		t.Fatalf("status code = %d, want %d; err=%v", got, http.StatusBadRequest, err)
	}
	assertCodexErrorCode(t, err.Error(), "invalid_request_error", "context_too_large")
	if !strings.Contains(err.Error(), "Your input exceeds the context window") {
		t.Fatalf("error message missing upstream context text: %v", err)
	}
}

func TestCodexExecutorExecuteStreamSurfacesTerminalStreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: response.created\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.5"}}` + "\n\n"))
		_, _ = w.Write([]byte("event: error\n"))
		_, _ = w.Write([]byte(`data: {"type":"error","error":{"type":"invalid_request_error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","param":"input"},"sequence_number":2}` + "\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.5",
		Payload: []byte(`{"model":"gpt-5.5","input":"hello"}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       true,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}

	var streamErr error
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			streamErr = chunk.Err
			break
		}
	}
	if streamErr == nil {
		t.Fatal("missing stream terminal error")
	}
	if got := statusCodeFromTestError(t, streamErr); got != http.StatusBadRequest {
		t.Fatalf("status code = %d, want %d; err=%v", got, http.StatusBadRequest, streamErr)
	}
	assertCodexErrorCode(t, streamErr.Error(), "invalid_request_error", "context_too_large")
}

func TestCodexTerminalStreamContextLengthErrFromResponseFailed(t *testing.T) {
	err, ok := codexTerminalStreamContextLengthErr([]byte(`{"type":"response.failed","response":{"id":"resp_1","status":"failed","error":{"code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again."}}}`))
	if !ok {
		t.Fatal("expected context length terminal error")
	}
	if got := statusCodeFromTestError(t, err); got != http.StatusBadRequest {
		t.Fatalf("status code = %d, want %d; err=%v", got, http.StatusBadRequest, err)
	}
	assertCodexErrorCode(t, err.Error(), "invalid_request_error", "context_too_large")
}

func TestCodexTerminalStreamContextLengthErrFromTopLevelError(t *testing.T) {
	err, ok := codexTerminalStreamContextLengthErr([]byte(`{"type":"error","code":"context_length_exceeded","message":"Your input exceeds the context window of this model. Please adjust your input and try again.","sequence_number":2}`))
	if !ok {
		t.Fatal("expected top-level context length terminal error")
	}
	if got := statusCodeFromTestError(t, err); got != http.StatusBadRequest {
		t.Fatalf("status code = %d, want %d; err=%v", got, http.StatusBadRequest, err)
	}
	assertCodexErrorCode(t, err.Error(), "invalid_request_error", "context_too_large")
	if !strings.Contains(err.Error(), "Your input exceeds the context window") {
		t.Fatalf("error message missing upstream context text: %v", err)
	}
}

func TestCodexTerminalStreamContextLengthErrIgnoresOtherTerminalErrors(t *testing.T) {
	_, ok := codexTerminalStreamContextLengthErr([]byte(`{"type":"error","error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"Rate limit reached."}}`))
	if ok {
		t.Fatal("rate limit terminal error should not be handled by context length fix")
	}
}

func TestCodexTerminalStreamErrIgnoresRateLimitTerminalErrors(t *testing.T) {
	_, _, ok := codexTerminalStreamErr([]byte(`{"type":"error","error":{"type":"rate_limit_error","code":"rate_limit_exceeded","message":"Rate limit reached."}}`))
	if ok {
		t.Fatal("rate limit terminal error should not be handled by replay terminal error path")
	}
}

func statusCodeFromTestError(t *testing.T, err error) int {
	t.Helper()

	statusErr, ok := err.(interface{ StatusCode() int })
	if !ok {
		t.Fatalf("error %T does not expose StatusCode(): %v", err, err)
	}
	return statusErr.StatusCode()
}

func TestCodexExecutorExecuteStream_EmptyStreamCompletionOutputUsesOutputItemDone(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]},\"output_index\":0}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"created_at\":1775555723,\"status\":\"completed\",\"model\":\"gpt-5.4-mini-2026-03-17\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":28,\"total_tokens\":36}}}\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5.4-mini",
		Payload: []byte(`{"model":"gpt-5.4-mini","input":"Say ok"}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       true,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}

	var completed []byte
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		payload := bytes.TrimSpace(chunk.Payload)
		if !bytes.HasPrefix(payload, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(payload[5:])
		if gjson.GetBytes(data, "type").String() == "response.completed" {
			completed = append([]byte(nil), data...)
		}
	}

	if len(completed) == 0 {
		t.Fatal("missing response.completed chunk")
	}

	gotContent := gjson.GetBytes(completed, "response.output.0.content.0.text").String()
	if gotContent != "ok" {
		t.Fatalf("response.output[0].content[0].text = %q, want %q; completed=%s", gotContent, "ok", string(completed))
	}
}

func TestCodexExecutorExecute_MergesAdjacentReasoningOutputItemsForClaude(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"resp_1","object":"response","created_at":1775555723,"status":"completed","model":"gpt-5","output":[{"id":"rs_1","type":"reasoning","encrypted_content":"sig_first","summary":[{"type":"summary_text","text":"First"}]},{"id":"rs_2","type":"reasoning","encrypted_content":"sig_second","summary":[{"type":"summary_text","text":"Second"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":8,"output_tokens":28,"total_tokens":36}}}` + "\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	resp, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude,
		Stream:       false,
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	thinkingCount := 0
	thinkingText := ""
	thinkingSignature := ""
	textFound := false
	for _, item := range gjson.GetBytes(resp.Payload, "content").Array() {
		switch item.Get("type").String() {
		case "thinking":
			thinkingCount++
			thinkingText = item.Get("thinking").String()
			thinkingSignature = item.Get("signature").String()
		case "text":
			if item.Get("text").String() == "done" {
				textFound = true
			}
		}
	}
	if thinkingCount != 1 {
		t.Fatalf("thinking block count = %d, want 1; payload=%s", thinkingCount, string(resp.Payload))
	}
	if thinkingText != "First\n\nSecond" {
		t.Fatalf("thinking text = %q, want %q; payload=%s", thinkingText, "First\n\nSecond", string(resp.Payload))
	}
	if thinkingSignature != "sig_second" {
		t.Fatalf("thinking signature = %q, want final signature; payload=%s", thinkingSignature, string(resp.Payload))
	}
	if !textFound {
		t.Fatalf("expected final text block after merged thinking; payload=%s", string(resp.Payload))
	}
}

func TestCodexExecutorExecuteStream_MergesConsecutiveReasoningItemsForClaude(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"sig_first\"}}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"First\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"sig_first\"},\"output_index\":0}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"sig_second\"}}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Second\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"sig_second\"},\"output_index\":1}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.content_part.added\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"done\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.content_part.done\"}\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"model\":\"gpt-5\",\"output\":[],\"usage\":{\"input_tokens\":8,\"output_tokens\":28,\"total_tokens\":36}}}\n\n"))
	}))
	defer server.Close()

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"base_url": server.URL,
		"api_key":  "test",
	}}

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatClaude,
		Stream:       true,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}

	thinkingIndexes := make(map[int64]bool)
	thinkingStartCount := 0
	thinkingStopCount := 0
	signatureCount := 0
	thinkingSignature := ""
	var thinkingDeltas []string
	textFound := false
	var allPayload bytes.Buffer
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
		allPayload.Write(chunk.Payload)
		for _, line := range strings.Split(string(chunk.Payload), "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			switch data.Get("type").String() {
			case "content_block_start":
				if data.Get("content_block.type").String() == "thinking" {
					thinkingStartCount++
					thinkingIndexes[data.Get("index").Int()] = true
				}
			case "content_block_delta":
				switch data.Get("delta.type").String() {
				case "thinking_delta":
					thinkingDeltas = append(thinkingDeltas, data.Get("delta.thinking").String())
				case "signature_delta":
					signatureCount++
					thinkingSignature = data.Get("delta.signature").String()
				case "text_delta":
					if data.Get("delta.text").String() == "done" {
						textFound = true
					}
				}
			case "content_block_stop":
				if thinkingIndexes[data.Get("index").Int()] {
					thinkingStopCount++
				}
			}
		}
	}

	if thinkingStartCount != 1 {
		t.Fatalf("thinking start count = %d, want 1; payload=%s", thinkingStartCount, allPayload.String())
	}
	if thinkingStopCount != 1 {
		t.Fatalf("thinking stop count = %d, want 1; payload=%s", thinkingStopCount, allPayload.String())
	}
	if got := strings.Join(thinkingDeltas, ""); got != "First\n\nSecond" {
		t.Fatalf("merged thinking text = %q, want %q; payload=%s", got, "First\n\nSecond", allPayload.String())
	}
	if signatureCount != 1 || thinkingSignature != "sig_second" {
		t.Fatalf("signature count/signature = %d/%q, want 1/final signature; payload=%s", signatureCount, thinkingSignature, allPayload.String())
	}
	if !textFound {
		t.Fatalf("expected final text delta after merged thinking; payload=%s", allPayload.String())
	}
}
