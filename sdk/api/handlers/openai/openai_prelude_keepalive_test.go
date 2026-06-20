package openai

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

func TestOpenAICompatPreludeErrorUsesOpenAISSEDataError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}, nil)
	h := NewOpenAIAPIHandler(base)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		t.Fatal("expected flusher")
	}
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage, 1)
	errs <- &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: errors.New("upstream unavailable")}
	close(errs)

	base.HandleStreamPrelude(c, flusher, func(error) {}, data, errs, nil, handlers.StreamPreludeOptions{
		CommitHeaders: func() {
			c.Header("Content-Type", "text/event-stream")
		},
		WriteFirstChunk: func(chunk []byte) {
			_, _ = c.Writer.Write([]byte("data: "))
			_, _ = c.Writer.Write(chunk)
			_, _ = c.Writer.Write([]byte("\n\n"))
		},
		WriteClosedBeforeData: func() {
			_, _ = c.Writer.Write([]byte("data: [DONE]\n\n"))
		},
		WritePreludeError: func(errMsg *interfaces.ErrorMessage) {
			status := http.StatusInternalServerError
			if errMsg != nil && errMsg.StatusCode > 0 {
				status = errMsg.StatusCode
			}
			errText := http.StatusText(status)
			if errMsg != nil && errMsg.Error != nil && errMsg.Error.Error() != "" {
				errText = errMsg.Error.Error()
			}
			body := handlers.BuildErrorResponseBody(status, errText)
			_, _ = c.Writer.Write([]byte("data: "))
			_, _ = c.Writer.Write(body)
			_, _ = c.Writer.Write([]byte("\n\n"))
		},
		Continue: func(data <-chan []byte, errs <-chan *interfaces.ErrorMessage) {
			h.handleStreamResult(c, flusher, func(error) {}, data, errs)
		},
	})

	body := recorder.Body.String()
	if !strings.Contains(body, "data: {") || !strings.Contains(body, `"error"`) || !strings.Contains(body, "upstream unavailable") {
		t.Fatalf("expected OpenAI SSE data error, got %q", body)
	}
}
