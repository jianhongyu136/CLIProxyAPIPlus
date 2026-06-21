package toolemu

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ToolSpec is a tool declaration normalized from upstream payload's `tools` array.
type ToolSpec struct {
	Name        string
	Description string
	SchemaJSON  json.RawMessage
}

// ToolChoiceKind enumerates the four supported tool_choice modes.
type ToolChoiceKind int

const (
	ToolChoiceKindAuto ToolChoiceKind = iota
	ToolChoiceKindNone
	ToolChoiceKindRequired
	ToolChoiceKindNamed
)

// ToolChoice is the locked-down tool_choice value used during rendering.
type ToolChoice struct {
	Kind            ToolChoiceKind
	Name            string // only for Kind == ToolChoiceKindNamed
	DisableParallel bool
}

// Convenience constructors.
var (
	ToolChoiceAuto     = ToolChoice{Kind: ToolChoiceKindAuto}
	ToolChoiceNone     = ToolChoice{Kind: ToolChoiceKindNone}
	ToolChoiceRequired = ToolChoice{Kind: ToolChoiceKindRequired}
)

// ToolChoiceNamed returns a Named tool_choice.
func ToolChoiceNamed(name string) ToolChoice {
	return ToolChoice{Kind: ToolChoiceKindNamed, Name: name}
}

const protocolFixedText = `<tool_protocol>
The ONLY accepted way to call a tool is to emit one or more blocks wrapped
EXACTLY in the literal ASCII tags <tool_call> and </tool_call>. No other
envelope, sentinel, or special token is recognized by this runtime.

Required shape:

<tool_call>
{"name": "<tool_name>", "arguments": { ...JSON args... }}
</tool_call>

Worked example (assuming a tool named get_weather is listed in <tools_doc>):

<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
</tool_call>

Rules:
- Output the entire tool_call block as raw text. Do NOT wrap it in markdown
  fences (no triple backticks), HTML, quotes, or any other container.
- Place natural-language text (if any) BEFORE the tool_call block(s), never
  inside or after the JSON object within a block.
- "name" field is REQUIRED and MUST exactly match one tool name from
  <tools_doc>. Use the bare name only: do NOT prefix it ("functions.",
  "tool.", "namespace."), do NOT suffix it with an index (":0", ":1"), do
  NOT translate or rename it.
- "arguments" MUST be a literal JSON object, even when empty: {}. Do NOT
  stringify it (no "arguments": "{...}"), do NOT omit the field.
- You MAY emit multiple <tool_call> blocks in one response; they will be
  executed in order.
- Do not invent tools that are not listed in <tools_doc>.

Formats that are NOT recognized and will be treated as plain text (your tool
call will silently fail). Do not emit any of these, even if your training
data suggests otherwise:
- Special-token envelopes such as <|tool_calls_section_begin|>,
  <|tool_call_begin|>, <|tool_call_argument_begin|>, <|tool_call_end|>,
  <|tool_calls_section_end|>, <|python_tag|>, or any other <|...|> sentinel.
- Provider-native call shapes: OpenAI streaming tool_call deltas, Anthropic
  <tool_use> blocks, Gemini functionCall objects, JSON-RPC frames,
  XML-style <function_call>/<invoke> tags.
- Markdown-fenced JSON (` + "```" + `json {...} ` + "```" + `) or a bare JSON object
  without the surrounding <tool_call> tags.

%s
</tool_protocol>`

// RenderInjection builds the deterministic injection text (tools_doc + tool_protocol).
// Inputs in any order produce the same output (R2). Output never contains
// volatile fields (no timestamps, request ids, nonces).
func RenderInjection(tools []ToolSpec, choice ToolChoice) string {
	sorted := append([]ToolSpec(nil), tools...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Name < sorted[j].Name })

	var b strings.Builder
	b.WriteString("<tools_doc>\n")
	toolsJSON := make([]any, 0, len(sorted))
	for _, t := range sorted {
		var schema any
		if err := json.Unmarshal([]byte(canonicalJSON(t.SchemaJSON)), &schema); err != nil {
			schema = map[string]any{}
		}
		toolsJSON = append(toolsJSON, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  schema,
			},
		})
	}
	toolsDoc, err := marshalSorted(toolsJSON)
	if err != nil {
		toolsDoc = []byte("[]")
	}
	b.Write(toolsDoc)
	b.WriteString("\n</tools_doc>\n")
	b.WriteString(fmt.Sprintf(protocolFixedText, renderToolChoiceClause(choice)))
	return b.String()
}

func renderToolChoiceClause(c ToolChoice) string {
	switch c.Kind {
	case ToolChoiceKindNone:
		return "You MUST NOT call any tool in this response. Answer in natural language only."
	case ToolChoiceKindRequired:
		return "You MUST call at least one tool in this response."
	case ToolChoiceKindNamed:
		return fmt.Sprintf(`You MUST call the tool named %q in this response.`, c.Name)
	default:
		return "You MAY call a tool when helpful, or answer directly without a tool call."
	}
}

// canonicalJSON re-serializes raw JSON with sorted object keys, no extra whitespace.
// Returns the input string verbatim on parse failure (defense in depth).
func canonicalJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	var v any
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return string(raw)
	}
	out, err := marshalSorted(v)
	if err != nil {
		return string(raw)
	}
	return string(out)
}

func marshalSorted(v any) ([]byte, error) {
	switch t := v.(type) {
	case json.RawMessage:
		// RawMessage is already canonical bytes; emit verbatim.
		return []byte(t), nil
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var b bytes.Buffer
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			kj, err := marshalLeafNoEscape(k)
			if err != nil {
				return nil, err
			}
			b.Write(kj)
			b.WriteByte(':')
			vj, err := marshalSorted(t[k])
			if err != nil {
				return nil, err
			}
			b.Write(vj)
		}
		b.WriteByte('}')
		return b.Bytes(), nil
	case []any:
		var b bytes.Buffer
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
			}
			ej, err := marshalSorted(e)
			if err != nil {
				return nil, err
			}
			b.Write(ej)
		}
		b.WriteByte(']')
		return b.Bytes(), nil
	default:
		return marshalLeafNoEscape(v)
	}
}

// marshalLeafNoEscape JSON-encodes a leaf value (string, number, bool, nil)
// without escaping `<`, `>`, `&` to their `\u00xx` forms. Protocol blocks such
// as `<tool_call>` must remain literal so the upstream model sees the same
// sentinel text after a JSON round-trip.
func marshalLeafNoEscape(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// TemplateVersion is bumped whenever the built-in template body changes; used
// by management status output to help debug cache-rate stepdowns after reloads.
const builtinTemplateVersion = 2

// TemplateVersion returns the current template version.
func TemplateVersion() int { return builtinTemplateVersion }
