package openai

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

func TestForwardResponsesStreamTerminalErrorUsesResponsesErrorChunk(t *testing.T) {
	gin.SetMode(gin.TestMode)
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, nil)
	h := NewOpenAIResponsesAPIHandler(base)

	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		t.Fatalf("expected gin writer to implement http.Flusher")
	}

	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage, 1)
	errs <- &interfaces.ErrorMessage{StatusCode: http.StatusInternalServerError, Error: errors.New("unexpected EOF")}
	close(errs)

	h.forwardResponsesStream(c, flusher, func(error) {}, data, errs, nil)
	body := recorder.Body.String()
	if !strings.Contains(body, `"type":"error"`) {
		t.Fatalf("expected responses error chunk, got: %q", body)
	}
	if strings.Contains(body, `"error":{`) {
		t.Fatalf("expected streaming error chunk (top-level type), got HTTP error body: %q", body)
	}
}

func TestResponsesPreludeErrorUsesResponsesSSEErrorEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}, nil)
	h := NewOpenAIResponsesAPIHandler(base)

	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		t.Fatalf("expected gin writer to implement http.Flusher")
	}

	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage, 1)
	errs <- &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: errors.New("responses upstream failed")}
	close(errs)
	framer := &responsesSSEFramer{}

	base.HandleStreamPrelude(c, flusher, func(error) {}, data, errs, nil, handlers.StreamPreludeOptions{
		CommitHeaders: func() {
			c.Header("Content-Type", "text/event-stream")
		},
		WriteFirstChunk: func(chunk []byte) {
			framer.WriteChunk(c.Writer, chunk)
		},
		WriteClosedBeforeData: func() {
			_, _ = c.Writer.Write([]byte("\n"))
		},
		WritePreludeError: func(errMsg *interfaces.ErrorMessage) {
			framer.Flush(c.Writer)
			status := http.StatusInternalServerError
			if errMsg != nil && errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			errText := http.StatusText(status)
			if errMsg != nil && errMsg.Error != nil && errMsg.Error.Error() != "" {
				errText = errMsg.Error.Error()
			}
			chunk := handlers.BuildOpenAIResponsesStreamErrorChunk(status, errText, 0)
			_, _ = fmt.Fprintf(c.Writer, "\nevent: error\ndata: %s\n\n", string(chunk))
		},
		Continue: func(data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
			h.forwardResponsesStream(c, flusher, func(error) {}, data, errs, framer)
		},
	})

	body := recorder.Body.String()
	if !strings.Contains(body, "event: error") || !strings.Contains(body, `"type":"error"`) || !strings.Contains(body, "responses upstream failed") {
		t.Fatalf("expected Responses SSE error event, got %q", body)
	}
}
