// Package toolemu emulates native tool-calling on upstream models that lack
// native support, by folding `tools`/`tool_choice`/historical tool-call
// messages into the system prompt and parsing structured `<tool_call>` JSON
// blocks out of the model's textual response.
package toolemu

// UpstreamShape identifies which upstream payload schema toolemu is operating on.
type UpstreamShape int

const (
	// ShapeOpenAIChat is the OpenAI chat-completions schema (messages/tools/tool_choice).
	ShapeOpenAIChat UpstreamShape = iota
	// ShapeOpenAIResponses is the OpenAI Responses schema (instructions/input/tools/tool_choice).
	ShapeOpenAIResponses
	// ShapeClaudeMessages is the Anthropic Claude messages schema.
	ShapeClaudeMessages
	// ShapeGeminiGenerateContent is the Gemini generateContent schema.
	ShapeGeminiGenerateContent
)

// ParsedToolCall is one structured tool call extracted from upstream assistant text.
type ParsedToolCall struct {
	ID        string
	Name      string
	Arguments []byte // canonical JSON object bytes (sorted keys)
}

// Parsed is the result of parsing an upstream assistant text response.
type Parsed struct {
	Prose     string
	ToolCalls []ParsedToolCall
}

// UpstreamMeta carries identifiers passed from the upstream response into id derivation
// and emitter construction.
type UpstreamMeta struct {
	Provider       string
	Model          string
	ResponseID     string
	Created        int64  // unix seconds, taken from upstream response
	UsagePayload   []byte // raw "usage" object from upstream, passed through verbatim
	FinishOverride string // if non-empty, force a finish reason; else "tool_calls" or "stop"
}

// FoldOpts controls FoldRequest behavior.
type FoldOpts struct {
	Shape    UpstreamShape
	Provider string
	Model    string
}
