package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

// StreamingPreludeKeepAliveEnabled reports whether handlers should emit SSE heartbeats before the first payload.
func StreamingPreludeKeepAliveEnabled(cfg *config.SDKConfig) bool {
	return cfg != nil && cfg.Streaming.PreludeKeepAlive && StreamingKeepAliveInterval(cfg) > 0
}

// StreamPreludeOptions provides protocol-specific behavior for the pre-first-payload stream phase.
type StreamPreludeOptions struct {
	CommitHeaders         func()
	WriteFirstChunk       func(chunk []byte)
	WriteClosedBeforeData func()
	WritePreludeError     func(errMsg *interfaces.ErrorMessage)
	WriteKeepAlive        func()
	Continue              func(data <-chan []byte, errs <-chan *interfaces.ErrorMessage)
}

// HandleStreamPrelude optionally commits SSE headers and emits heartbeats while waiting for the first stream payload.
// It returns true when the prelude path handled the request. When it returns false, callers should use their legacy
// first-chunk waiting logic.
func (h *BaseAPIHandler) HandleStreamPrelude(c *gin.Context, flusher http.Flusher, cancel func(error), data <-chan []byte, errs <-chan *interfaces.ErrorMessage, _ http.Header, opts StreamPreludeOptions) bool {
	if h == nil || c == nil || flusher == nil || cancel == nil || !StreamingPreludeKeepAliveEnabled(h.Cfg) {
		return false
	}
	if opts.CommitHeaders == nil || opts.WriteFirstChunk == nil || opts.WriteClosedBeforeData == nil || opts.WritePreludeError == nil || opts.Continue == nil {
		return false
	}

	writeKeepAlive := opts.WriteKeepAlive
	if writeKeepAlive == nil {
		writeKeepAlive = func() {
			_, _ = c.Writer.Write([]byte(": keep-alive\n\n"))
		}
	}

	opts.CommitHeaders()
	flusher.Flush()

	interval := StreamingKeepAliveInterval(h.Cfg)
	keepAlive := time.NewTicker(interval)
	defer keepAlive.Stop()

	writePreludeError := func(errMsg *interfaces.ErrorMessage) {
		opts.WritePreludeError(errMsg)
		flusher.Flush()
		if errMsg != nil {
			cancel(errMsg.Error)
		} else {
			cancel(nil)
		}
	}

	for {
		select {
		case <-c.Request.Context().Done():
			cancel(c.Request.Context().Err())
			return true
		case errMsg, ok := <-errs:
			if !ok {
				errs = nil
				continue
			}
			writePreludeError(errMsg)
			return true
		case chunk, ok := <-data:
			if !ok {
				if errMsg, okPendingErr := pendingStreamPreludeError(errs); okPendingErr {
					writePreludeError(errMsg)
					return true
				}
				opts.WriteClosedBeforeData()
				flusher.Flush()
				cancel(nil)
				return true
			}
			opts.WriteFirstChunk(chunk)
			flusher.Flush()
			opts.Continue(data, errs)
			return true
		case <-keepAlive.C:
			writeKeepAlive()
			flusher.Flush()
		}
	}
}

func pendingStreamPreludeError(errs <-chan *interfaces.ErrorMessage) (*interfaces.ErrorMessage, bool) {
	if errs == nil {
		return nil, false
	}
	select {
	case errMsg, ok := <-errs:
		if !ok || errMsg == nil {
			return nil, false
		}
		return errMsg, true
	default:
		return nil, false
	}
}
