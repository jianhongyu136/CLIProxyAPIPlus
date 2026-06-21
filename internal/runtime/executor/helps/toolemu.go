package helps

import (
	"context"
	"fmt"
	"io"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/toolemu"
)

// ToolEmuActive reports whether toolemu should activate for the given request.
// Wraps the toolemu.Default matcher + HasToolArtifacts probe so executors don't
// need to import the toolemu package solely for the activation check. Emits an
// info log on each activation so operators can confirm toolemu took over the
// request — including subsequent turns where the client only sends tool_calls /
// role=tool history without redeclaring tools.
func ToolEmuActive(ctx context.Context, provider, model, alias string, payload []byte) bool {
	if !toolemu.Default.IsEnabled(provider, model, alias) {
		return false
	}
	if !toolemu.HasToolArtifacts(payload) {
		return false
	}
	LogWithRequestID(ctx).Infof("toolemu: activated for provider=%s model=%s alias=%s", provider, model, alias)
	return true
}

// ToolEmuOutcome bundles the fold→parse loop result with derived response artefacts.
// FakeStreamChunks is populated for streaming callers; BuiltBody is populated for
// non-streaming callers. Folded is the upstream payload actually sent (post-fold).
type ToolEmuOutcome struct {
	Folded           []byte
	BuiltBody        []byte
	FakeStreamChunks [][]byte
	Result           toolemu.ParseResult
}

// RunToolEmu performs the full fold → send → parse → retry → build cycle for a
// single upstream request. The caller supplies a send closure that executes
// the actual HTTP POST against the upstream and returns the response body.
//
// For each shape the helper also constructs the corresponding upstream-format
// non-stream body (BuiltBody) and fake-stream chunks (FakeStreamChunks) so
// callers can choose the path without re-running the build step.
func RunToolEmu(
	ctx context.Context,
	payload []byte,
	shape toolemu.UpstreamShape,
	provider string,
	policy toolemu.RetryPolicy,
	send toolemu.UpstreamSendFunc,
) (ToolEmuOutcome, error) {
	choice := toolemu.ExtractToolChoice(payload, shape)
	folded, err := toolemu.FoldRequest(payload, toolemu.FoldOpts{Shape: shape, Provider: provider})
	if err != nil {
		return ToolEmuOutcome{}, err
	}
	res, err := toolemu.ParseAndRetry(ctx, folded, send, shape, policy, choice)
	if err != nil {
		return ToolEmuOutcome{Folded: folded}, err
	}
	res.Meta.Provider = provider

	out := ToolEmuOutcome{Folded: folded, Result: res}
	switch shape {
	case toolemu.ShapeOpenAIChat:
		out.BuiltBody, _ = toolemu.BuildChatCompletion(res.Parsed, res.Meta)
		out.FakeStreamChunks, _ = toolemu.FakeStreamChat(res.Parsed, res.Meta)
	case toolemu.ShapeOpenAIResponses:
		out.BuiltBody, _ = toolemu.BuildResponses(res.Parsed, res.Meta)
		out.FakeStreamChunks, _ = toolemu.FakeStreamResponses(res.Parsed, res.Meta)
	case toolemu.ShapeClaudeMessages:
		out.BuiltBody, _ = toolemu.BuildClaudeMessage(res.Parsed, res.Meta)
	case toolemu.ShapeGeminiGenerateContent:
		out.BuiltBody, _ = toolemu.BuildGeminiGenerateContent(res.Parsed, res.Meta)
	}
	return out, nil
}

func RunToolEmuStream(ctx context.Context, meta toolemu.UpstreamMeta, shape toolemu.UpstreamShape, upstreamResp io.Reader, choice toolemu.ToolChoice, onFrame func([]byte)) (toolemu.UpstreamMeta, error) {
	if upstreamResp == nil {
		return toolemu.UpstreamMeta{}, fmt.Errorf("toolemu stream: nil upstream response")
	}
	if meta.Provider == "" {
		meta.Provider = "unknown"
	}
	var events toolemu.StreamEvents
	switch shape {
	case toolemu.ShapeOpenAIChat:
		emitter := toolemu.NewChatStreamEmitter(meta, onFrame)
		emitter.SetToolChoice(choice)
		events = emitter.Events()
	case toolemu.ShapeOpenAIResponses:
		emitter := toolemu.NewResponsesStreamEmitter(meta, onFrame)
		emitter.SetToolChoice(choice)
		events = emitter.Events()
	case toolemu.ShapeClaudeMessages:
		emitter := toolemu.NewClaudeStreamEmitter(meta, onFrame)
		emitter.SetToolChoice(choice)
		events = emitter.Events()
	default:
		return toolemu.UpstreamMeta{}, fmt.Errorf("toolemu stream: unsupported shape %d", shape)
	}
	parser := toolemu.NewStreamParser(events, meta)
	pump := toolemu.UpstreamPump{Reader: upstreamResp, Shape: shape, Parser: parser}
	upMeta, err := pump.Run(ctx)
	if upMeta.Provider == "" {
		upMeta.Provider = meta.Provider
	}
	if upMeta.Model == "" {
		upMeta.Model = meta.Model
	}
	if upMeta.ResponseID == "" {
		upMeta.ResponseID = meta.ResponseID
	}
	if upMeta.Created == 0 {
		upMeta.Created = meta.Created
	}
	return upMeta, err
}

// ToolEmuRetryPolicy resolves a runtime retry policy from a config snapshot.
// When ParseRetry is zero and OnParseFailure is unset, DefaultParseRetry is
// applied so opt-in users get the documented "one extra attempt" behavior.
func ToolEmuRetryPolicy(cfg toolemu.ToolEmulationConfig) toolemu.RetryPolicy {
	c := cfg.DefaultsApplied()
	attempts := c.ParseRetry
	if attempts == 0 && cfg.ParseRetry == 0 && cfg.OnParseFailure == "" {
		attempts = toolemu.DefaultParseRetry
	}
	return toolemu.RetryPolicy{Attempts: attempts, OnFailure: c.OnParseFailure}
}
