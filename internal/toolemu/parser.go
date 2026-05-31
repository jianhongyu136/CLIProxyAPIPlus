package toolemu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	log "github.com/sirupsen/logrus"
	"github.com/tailscale/hujson"
)

const (
	openTag  = "<tool_call>"
	closeTag = "</tool_call>"
)

// Parse extracts <tool_call>{...}</tool_call> blocks from upstream assistant
// text into a structured Parsed result. The tool_call body is extracted via
// JSON brace tracking so a closing tag substring embedded inside a JSON
// string literal (common in code-generation tools) does not cause a
// mid-payload split. Sentinel matching is case-insensitive so common model
// deviations such as <Tool_Call> or <TOOL_CALL> still parse. Unterminated
// tags and unrecoverably malformed JSON produce errors so the caller can
// drive ParseAndRetry / degrade policy.
func Parse(text string) (Parsed, error) {
	var proseParts []string
	var calls []ParsedToolCall
	validOpenTags := unescapedToolCallOpenOffsets(text)
	i := 0
	for i < len(text) {
		openAt := nextOpenTagOffset(validOpenTags, i)
		if openAt < 0 {
			proseParts = append(proseParts, text[i:])
			break
		}
		bodyStart, bodyEnd, nextStart, ok := locateToolCallBlock(text, openAt)
		if !ok {
			return Parsed{}, fmt.Errorf("toolemu: unterminated <tool_call>")
		}
		proseParts = append(proseParts, text[i:openAt])
		payload := strings.TrimSpace(text[bodyStart:bodyEnd])
		call, err := parseOneCall(payload)
		if err != nil {
			// TODO: Remove or redact raw_text after the debugging window closes.
			log.WithError(err).WithField("raw_text", text).Error("tool call parse failed")
			return Parsed{}, err
		}
		calls = append(calls, call)
		i = nextStart
	}
	prose := strings.TrimRight(strings.TrimLeft(strings.Join(proseParts, ""), "\n"), "\n")
	return Parsed{Prose: prose, ToolCalls: calls}, nil
}

func unescapedToolCallOpenOffsets(text string) []int {
	var offsets []int
	md := newMdContext()
	for i, r := range text {
		md.feedRune(r)
		if r == '<' && !md.inEscapedContext() && i+len(openTag) <= len(text) && strings.EqualFold(text[i:i+len(openTag)], openTag) {
			offsets = append(offsets, i)
		}
	}
	return offsets
}

func nextOpenTagOffset(offsets []int, start int) int {
	for _, offset := range offsets {
		if offset >= start {
			return offset
		}
	}
	return -1
}

// locateToolCallBlock resolves the JSON body range for a <tool_call> block
// whose opening tag starts at openAt. The primary path uses a JSON brace
// tracker that respects string literals — this prevents a sentinel substring
// embedded in arguments (e.g. {"code":"...</tool_call>..."}) from prematurely
// closing the block. The optional trailing </tool_call> tag is consumed when
// present but not required, so models that omit the closing tag after a
// complete JSON object still parse.
//
// If the brace tracker cannot find a balanced object (e.g. body is not a
// JSON object), the function falls back to a literal closing-tag search to
// preserve historical behavior for non-JSON bodies (which will fail in
// parseOneCall anyway, but with a clearer error path).
func locateToolCallBlock(text string, openAt int) (bodyStart, bodyEnd, nextStart int, ok bool) {
	bodyStart = openAt + len(openTag)
	p := bodyStart
	for p < len(text) && isASCIISpace(text[p]) {
		p++
	}
	if p < len(text) && text[p] == '{' {
		if endJSON, jsonOK := findJSONObjectEnd(text, p); jsonOK {
			tail := endJSON
			for tail < len(text) && isASCIISpace(text[tail]) {
				tail++
			}
			if tail+len(closeTag) <= len(text) && strings.EqualFold(text[tail:tail+len(closeTag)], closeTag) {
				return bodyStart, endJSON, tail + len(closeTag), true
			}
			// Closing tag missing — accept the block anyway and resume right
			// after the JSON object so subsequent prose is preserved.
			return bodyStart, endJSON, endJSON, true
		}
	}
	// Fallback: literal closing-tag search. Used when the body is not a JSON
	// object (e.g. degraded models emit prose inside <tool_call>); parseOneCall
	// will then surface a clean JSON error.
	closeAt := caseInsensitiveIndex(text, closeTag, bodyStart)
	if closeAt < 0 {
		return 0, 0, 0, false
	}
	return bodyStart, closeAt, closeAt + len(closeTag), true
}

// findJSONObjectEnd returns the byte offset immediately after the '}' that
// closes the JSON object beginning at text[start] (which must be '{'). String
// literals and escape sequences are tracked so braces appearing inside
// strings do not affect depth. Returns ok=false if the object is unterminated
// before end of text or if depth goes negative.
func findJSONObjectEnd(text string, start int) (int, bool) {
	if start >= len(text) || text[start] != '{' {
		return 0, false
	}
	depth := 0
	inStr := false
	escape := false
	for i := start; i < len(text); i++ {
		c := text[i]
		if inStr {
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return i + 1, true
			}
			if depth < 0 {
				return 0, false
			}
		}
	}
	return 0, false
}

// caseInsensitiveIndex returns the byte offset of needle (matched with
// strings.EqualFold semantics) at or after start, or -1 if not present.
func caseInsensitiveIndex(haystack, needle string, start int) int {
	if needle == "" {
		return start
	}
	n := len(needle)
	for i := start; i+n <= len(haystack); i++ {
		if strings.EqualFold(haystack[i:i+n], needle) {
			return i
		}
	}
	return -1
}

func isASCIISpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func parseOneCall(payload string) (ParsedToolCall, error) {
	obj, err := tryParseJSONObject([]byte(payload))
	if err != nil {
		return ParsedToolCall{}, fmt.Errorf("toolemu: invalid JSON inside <tool_call>: %w", err)
	}

	nameRaw, ok := obj["name"]
	if !ok {
		return ParsedToolCall{}, fmt.Errorf("toolemu: <tool_call> missing name")
	}
	var name string
	if errName := json.Unmarshal(nameRaw, &name); errName != nil {
		return ParsedToolCall{}, fmt.Errorf("toolemu: <tool_call> name not a string")
	}
	name = normalizeToolName(name)
	if name == "" {
		return ParsedToolCall{}, fmt.Errorf("toolemu: <tool_call> name is empty")
	}

	id := ""
	if raw, ok := obj["id"]; ok {
		_ = json.Unmarshal(raw, &id)
	}

	args, err := normalizeArguments(obj["arguments"])
	if err != nil {
		return ParsedToolCall{}, err
	}
	return ParsedToolCall{ID: id, Name: name, Arguments: args}, nil
}

// tryParseJSONObject runs progressively-lenient passes until one succeeds:
//  1. strict encoding/json.Unmarshal
//  2. strip trailing commas (string-aware) + retry
//  3. hujson.Standardize (handles comments + trailing commas + extra whitespace) + retry
//
// Failures here drive the retry-or-degrade loop in retry.go so this function
// stays the single chokepoint for tolerance policy.
func tryParseJSONObject(b []byte) (map[string]json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err == nil {
		return obj, nil
	}
	if repaired := stripTrailingCommas(b); !bytes.Equal(repaired, b) {
		if err := json.Unmarshal(repaired, &obj); err == nil {
			return obj, nil
		}
	}
	if std, errHu := hujson.Standardize(b); errHu == nil {
		if err := json.Unmarshal(std, &obj); err == nil {
			return obj, nil
		}
	}
	return nil, fmt.Errorf("invalid JSON object")
}

// stripTrailingCommas drops commas that immediately precede a closing `}` or
// `]`, skipping over string literals so commas inside strings are preserved.
// Comments are not supported here (hujson handles those in the next pass).
func stripTrailingCommas(b []byte) []byte {
	out := make([]byte, 0, len(b))
	inStr := false
	escape := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		if inStr {
			out = append(out, c)
			if escape {
				escape = false
				continue
			}
			if c == '\\' {
				escape = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}
		if c == '"' {
			inStr = true
			out = append(out, c)
			continue
		}
		if c == ',' {
			j := i + 1
			for j < len(b) && (b[j] == ' ' || b[j] == '\t' || b[j] == '\n' || b[j] == '\r') {
				j++
			}
			if j < len(b) && (b[j] == '}' || b[j] == ']') {
				continue
			}
		}
		out = append(out, c)
	}
	return out
}

// normalizeArguments converts the upstream "arguments" field to canonical
// sorted JSON object bytes. It tolerates the most common model deviations:
//   - missing entry, JSON null, or empty/whitespace string  → "{}"
//   - object value                                          → canonical form
//   - string-encoded JSON object (with optional trailing commas/comments) → canonical form
//
// Anything that still cannot be coerced into a JSON object returns an error.
func normalizeArguments(raw json.RawMessage) ([]byte, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return []byte("{}"), nil
	}
	if bytes.Equal(trimmed, []byte(`""`)) {
		return []byte("{}"), nil
	}

	if canon, ok := tryCanonicalObject(trimmed); ok {
		return canon, nil
	}

	var argsStr string
	if err := json.Unmarshal(trimmed, &argsStr); err == nil {
		s := strings.TrimSpace(argsStr)
		if s == "" || s == "null" {
			return []byte("{}"), nil
		}
		if canon, ok := tryCanonicalObject([]byte(s)); ok {
			return canon, nil
		}
		return nil, fmt.Errorf("toolemu: <tool_call> arguments string is not a JSON object")
	}
	return nil, fmt.Errorf("toolemu: <tool_call> arguments not a JSON object")
}

// tryCanonicalObject unmarshals b as a JSON object, applying the same lenient
// cascade as tryParseJSONObject (strict → trailing-comma repair → hujson).
// On success the object is re-emitted via marshalSorted to keep arguments
// byte-stable (R3).
func tryCanonicalObject(b []byte) ([]byte, bool) {
	var obj map[string]any
	if err := json.Unmarshal(b, &obj); err == nil {
		canon, _ := marshalSorted(obj)
		return canon, true
	}
	if repaired := stripTrailingCommas(b); !bytes.Equal(repaired, b) {
		if err := json.Unmarshal(repaired, &obj); err == nil {
			canon, _ := marshalSorted(obj)
			return canon, true
		}
	}
	if std, err := hujson.Standardize(b); err == nil {
		if err := json.Unmarshal(std, &obj); err == nil {
			canon, _ := marshalSorted(obj)
			return canon, true
		}
	}
	return nil, false
}

// normalizeToolName strips well-known synthetic namespace prefixes that some
// upstream models prepend (e.g. "functions.get_weather", "tools.get_weather").
// Only the documented OpenAI/Anthropic-style prefixes are stripped so a real
// namespaced tool like "vendor.do_thing" is preserved verbatim.
func normalizeToolName(name string) string {
	name = strings.TrimSpace(name)
	lower := strings.ToLower(name)
	for _, prefix := range []string{"functions.", "tools.", "function.", "tool."} {
		if strings.HasPrefix(lower, prefix) {
			return strings.TrimSpace(name[len(prefix):])
		}
	}
	return name
}
