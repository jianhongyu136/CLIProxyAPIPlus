package toolemu

import (
	"context"
	"fmt"

	"github.com/tidwall/sjson"
)

// RetryPolicy configures ParseAndRetry behavior.
type RetryPolicy struct {
	Attempts  int    // additional attempts after the first; 0 = no retry
	OnFailure string // "parse_failed_to_content" or "error"
}

// UpstreamSendFunc sends a (possibly retried) payload to upstream and returns
// the non-streaming response body. Errors from this function propagate as-is
// (they are not parse failures — credential or transport errors).
type UpstreamSendFunc func(ctx context.Context, payload []byte) ([]byte, error)

// ParseResult is what ParseAndRetry returns on success or degrade.
type ParseResult struct {
	Parsed   Parsed
	Meta     UpstreamMeta
	Degraded bool // true when parsing failed and we returned raw text as prose
}

// ParseFailedError is returned when OnFailure=="error" and all attempts fail.
type ParseFailedError struct {
	LastText string
}

func (e ParseFailedError) Error() string { return "toolemu: parse failed after retries" }

// ToolChoiceValidationError reports a parsed response that violates the effective tool_choice.
type ToolChoiceValidationError struct {
	Choice ToolChoice
	Parsed Parsed
}

func (e ToolChoiceValidationError) Error() string {
	if e.Choice.DisableParallel && len(e.Parsed.ToolCalls) > 1 {
		return "toolemu: tool_choice disable_parallel_tool_use forbids multiple tool calls"
	}
	switch e.Choice.Kind {
	case ToolChoiceKindNone:
		return "toolemu: tool_choice none forbids tool calls"
	case ToolChoiceKindRequired:
		return "toolemu: tool_choice required needs at least one tool call"
	case ToolChoiceKindNamed:
		return fmt.Sprintf("toolemu: tool_choice named requires only %q", e.Choice.Name)
	default:
		return "toolemu: tool_choice validation failed"
	}
}

// ValidateToolChoice enforces native tool_choice semantics on parsed emulated output.
func ValidateToolChoice(parsed Parsed, choice ToolChoice) error {
	switch choice.Kind {
	case ToolChoiceKindNone:
		if len(parsed.ToolCalls) > 0 {
			return ToolChoiceValidationError{Choice: choice, Parsed: parsed}
		}
	case ToolChoiceKindRequired:
		if len(parsed.ToolCalls) == 0 {
			return ToolChoiceValidationError{Choice: choice, Parsed: parsed}
		}
	case ToolChoiceKindNamed:
		if len(parsed.ToolCalls) == 0 {
			return ToolChoiceValidationError{Choice: choice, Parsed: parsed}
		}
		for _, call := range parsed.ToolCalls {
			if call.Name != choice.Name {
				return ToolChoiceValidationError{Choice: choice, Parsed: parsed}
			}
		}
	}
	if choice.DisableParallel && len(parsed.ToolCalls) > 1 {
		return ToolChoiceValidationError{Choice: choice, Parsed: parsed}
	}
	return nil
}

const retryInstruction = "Your previous response did not contain a valid <tool_call> block (or contained malformed JSON). Please re-emit your response strictly following the <tool_protocol> rules in the system prompt."

// ParseAndRetry runs the fold→send→parse loop with optional soft retry.
// On final parse failure, behavior is governed by policy.OnFailure.
func ParseAndRetry(ctx context.Context, payload []byte, send UpstreamSendFunc, shape UpstreamShape, policy RetryPolicy, choice ToolChoice) (ParseResult, error) {
	current := payload
	var lastText string
	var lastMeta UpstreamMeta
	var lastErr error
	maxAttempts := 1 + policy.Attempts
	for attempt := 0; attempt < maxAttempts; attempt++ {
		body, err := send(ctx, current)
		if err != nil {
			return ParseResult{}, err
		}
		text, meta, _ := ExtractAssistantText(body, shape)
		lastText, lastMeta = text, meta
		parsed, perr := Parse(text)
		if perr == nil {
			perr = ValidateToolChoice(parsed, choice)
		}
		if perr == nil {
			return ParseResult{Parsed: parsed, Meta: meta}, nil
		}
		lastErr = perr
		if attempt == maxAttempts-1 {
			break
		}
		retryPayload, rerr := appendRetryUserMessage(current, shape)
		if rerr != nil {
			return ParseResult{}, rerr
		}
		current = retryPayload
	}

	if _, ok := lastErr.(ToolChoiceValidationError); ok {
		return ParseResult{}, lastErr
	}
	switch policy.OnFailure {
	case "error":
		return ParseResult{}, ParseFailedError{LastText: lastText}
	default: // parse_failed_to_content
		return ParseResult{
			Parsed:   Parsed{Prose: lastText},
			Meta:     lastMeta,
			Degraded: true,
		}, nil
	}
}

func appendRetryUserMessage(payload []byte, shape UpstreamShape) ([]byte, error) {
	switch shape {
	case ShapeOpenAIChat:
		msg := map[string]any{"role": "user", "content": retryInstruction}
		raw, err := marshalSorted(msg)
		if err != nil {
			return nil, fmt.Errorf("toolemu: marshal retry user message: %w", err)
		}
		out, err := sjson.SetRawBytes(payload, "messages.-1", raw)
		if err != nil {
			return nil, fmt.Errorf("toolemu: append retry user message: %w", err)
		}
		return out, nil
	case ShapeOpenAIResponses:
		item := map[string]any{
			"type": "message", "role": "user",
			"content": []any{map[string]any{"type": "input_text", "text": retryInstruction}},
		}
		raw, err := marshalSorted(item)
		if err != nil {
			return nil, fmt.Errorf("toolemu: marshal retry user message: %w", err)
		}
		out, err := sjson.SetRawBytes(payload, "input.-1", raw)
		if err != nil {
			return nil, fmt.Errorf("toolemu: append retry user message: %w", err)
		}
		return out, nil
	case ShapeClaudeMessages:
		msg := map[string]any{
			"role":    "user",
			"content": []any{map[string]any{"type": "text", "text": retryInstruction}},
		}
		raw, err := marshalSorted(msg)
		if err != nil {
			return nil, fmt.Errorf("toolemu: marshal retry user message: %w", err)
		}
		out, err := sjson.SetRawBytes(payload, "messages.-1", raw)
		if err != nil {
			return nil, fmt.Errorf("toolemu: append retry user message: %w", err)
		}
		return out, nil
	case ShapeGeminiGenerateContent:
		msg := map[string]any{
			"role":  "user",
			"parts": []any{map[string]any{"text": retryInstruction}},
		}
		raw, err := marshalSorted(msg)
		if err != nil {
			return nil, fmt.Errorf("toolemu: marshal retry user message: %w", err)
		}
		out, err := sjson.SetRawBytes(payload, "contents.-1", raw)
		if err != nil {
			return nil, fmt.Errorf("toolemu: append retry user message: %w", err)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("toolemu: unknown shape %d", shape)
	}
}
