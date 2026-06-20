package handlers

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

func newPreludeTestContext(t *testing.T) (*httptest.ResponseRecorder, *gin.Context, http.Flusher) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/test", nil)
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		t.Fatalf("expected gin writer to implement http.Flusher")
	}
	return recorder, c, flusher
}

func TestStreamingPreludeKeepAliveEnabledRequiresSwitchAndInterval(t *testing.T) {
	if StreamingPreludeKeepAliveEnabled(nil) {
		t.Fatal("nil config should disable prelude keepalive")
	}
	if StreamingPreludeKeepAliveEnabled(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{PreludeKeepAlive: true}}) {
		t.Fatal("prelude keepalive should require a positive interval")
	}
	if StreamingPreludeKeepAliveEnabled(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1}}) {
		t.Fatal("prelude keepalive should require the explicit switch")
	}
	if !StreamingPreludeKeepAliveEnabled(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}) {
		t.Fatal("prelude keepalive should be enabled when switch and interval are set")
	}
}

func TestHandleStreamPreludeEmitsKeepAliveBeforeFirstChunk(t *testing.T) {
	base := NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}, nil)
	recorder, c, flusher := newPreludeTestContext(t)
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage)
	continued := make(chan struct{}, 1)
	cancelErrs := make(chan error, 1)

	go func() {
		time.Sleep(1100 * time.Millisecond)
		data <- []byte(`{"ok":true}`)
		close(data)
		close(errs)
	}()

	handled := base.HandleStreamPrelude(c, flusher, func(err error) { cancelErrs <- err }, data, errs, http.Header{"X-Upstream": []string{"yes"}}, StreamPreludeOptions{
		CommitHeaders: func() {
			c.Header("Content-Type", "text/event-stream")
			WriteUpstreamHeaders(c.Writer.Header(), http.Header{"X-Upstream": []string{"yes"}})
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
			_, _ = c.Writer.Write([]byte("event: error\n\n"))
		},
		Continue: func(<-chan []byte, <-chan *interfaces.ErrorMessage) {
			continued <- struct{}{}
		},
	})

	if !handled {
		t.Fatal("expected prelude helper to handle enabled config")
	}
	body := recorder.Body.String()
	if !strings.Contains(body, ": keep-alive\n\n") {
		t.Fatalf("expected prelude keepalive before first chunk, got %q", body)
	}
	if !strings.Contains(body, "data: {\"ok\":true}\n\n") {
		t.Fatalf("expected first chunk to be written, got %q", body)
	}
	if recorder.Header().Get("X-Upstream") != "yes" {
		t.Fatalf("expected upstream header passthrough, got %q", recorder.Header().Get("X-Upstream"))
	}
	select {
	case <-continued:
	default:
		t.Fatal("expected Continue callback after first chunk")
	}
}

func TestHandleStreamPreludeWritesSSEErrorAfterHeadersCommitted(t *testing.T) {
	base := NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}, nil)
	recorder, c, flusher := newPreludeTestContext(t)
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage, 1)
	boom := errors.New("upstream failed before first payload")
	errs <- &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: boom}
	close(errs)
	cancelErrs := make(chan error, 1)

	handled := base.HandleStreamPrelude(c, flusher, func(err error) { cancelErrs <- err }, data, errs, nil, StreamPreludeOptions{
		CommitHeaders: func() {
			c.Header("Content-Type", "text/event-stream")
		},
		WriteFirstChunk: func([]byte) {
			t.Fatal("first chunk should not be written")
		},
		WriteClosedBeforeData: func() {
			t.Fatal("closed marker should not be written")
		},
		WritePreludeError: func(errMsg *interfaces.ErrorMessage) {
			_, _ = c.Writer.Write([]byte("event: error\n"))
			_, _ = c.Writer.Write([]byte("data: failed\n\n"))
		},
		Continue: func(<-chan []byte, <-chan *interfaces.ErrorMessage) {
			t.Fatal("stream should not continue after prelude error")
		},
	})

	if !handled {
		t.Fatal("expected prelude helper to handle enabled config")
	}
	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", got)
	}
	if body := recorder.Body.String(); !strings.Contains(body, "event: error") || !strings.Contains(body, "data: failed") {
		t.Fatalf("expected SSE error body, got %q", body)
	}
	select {
	case got := <-cancelErrs:
		if !errors.Is(got, boom) {
			t.Fatalf("cancel error = %v, want %v", got, boom)
		}
	default:
		t.Fatal("expected cancel callback with upstream error")
	}
}

func TestHandleStreamPreludePrefersPendingErrorWhenDataCloses(t *testing.T) {
	for i := 0; i < 100; i++ {
		base := NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1, PreludeKeepAlive: true}}, nil)
		recorder, c, flusher := newPreludeTestContext(t)
		data := make(chan []byte)
		close(data)
		errs := make(chan *interfaces.ErrorMessage, 1)
		boom := errors.New("upstream failed while closing before first payload")
		errs <- &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: boom}
		close(errs)
		cancelErrs := make(chan error, 1)

		handled := base.HandleStreamPrelude(c, flusher, func(err error) { cancelErrs <- err }, data, errs, nil, StreamPreludeOptions{
			CommitHeaders: func() {
				c.Header("Content-Type", "text/event-stream")
			},
			WriteFirstChunk: func([]byte) {
				t.Fatal("first chunk should not be written")
			},
			WriteClosedBeforeData: func() {
				_, _ = c.Writer.Write([]byte("data: [DONE]\n\n"))
			},
			WritePreludeError: func(errMsg *interfaces.ErrorMessage) {
				_, _ = c.Writer.Write([]byte("event: error\n"))
				_, _ = c.Writer.Write([]byte("data: failed\n\n"))
			},
			Continue: func(<-chan []byte, <-chan *interfaces.ErrorMessage) {
				t.Fatal("stream should not continue after prelude close")
			},
		})

		if !handled {
			t.Fatal("expected prelude helper to handle enabled config")
		}
		body := recorder.Body.String()
		if strings.Contains(body, "data: [DONE]") {
			t.Fatalf("iteration %d wrote closed marker instead of pending error: %q", i, body)
		}
		if !strings.Contains(body, "event: error") || !strings.Contains(body, "data: failed") {
			t.Fatalf("iteration %d expected pending SSE error, got %q", i, body)
		}
		select {
		case got := <-cancelErrs:
			if !errors.Is(got, boom) {
				t.Fatalf("cancel error = %v, want %v", got, boom)
			}
		default:
			t.Fatal("expected cancel callback with upstream error")
		}
	}
}

func TestHandleStreamPreludeReturnsFalseWhenDisabled(t *testing.T) {
	base := NewBaseAPIHandlers(&sdkconfig.SDKConfig{Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1}}, nil)
	_, c, flusher := newPreludeTestContext(t)
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage)

	handled := base.HandleStreamPrelude(c, flusher, func(error) {}, data, errs, nil, StreamPreludeOptions{
		CommitHeaders:         func() { t.Fatal("headers should not be committed when disabled") },
		WriteFirstChunk:       func([]byte) { t.Fatal("first chunk should not be written when disabled") },
		WriteClosedBeforeData: func() { t.Fatal("closed marker should not be written when disabled") },
		WritePreludeError:     func(*interfaces.ErrorMessage) { t.Fatal("error should not be written when disabled") },
		Continue: func(<-chan []byte, <-chan *interfaces.ErrorMessage) {
			t.Fatal("stream should not continue when disabled")
		},
	})
	if handled {
		t.Fatal("expected disabled prelude helper to return false")
	}
}
