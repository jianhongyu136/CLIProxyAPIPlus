package toolemu

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// FoldRequest is the canonical request-side entry point. Given an upstream
// payload (OpenAI chat-completions or Responses), it folds tools/tool_choice
// and historical tool messages into prompt text and returns the new payload.
//
// The original payload is not mutated; a new byte slice is returned. tools and
// tool_choice fields are removed from the upstream payload.
func FoldRequest(payload []byte, opts FoldOpts) ([]byte, error) {
	switch opts.Shape {
	case ShapeOpenAIChat:
		return foldChat(payload)
	case ShapeOpenAIResponses:
		return foldResponses(payload)
	case ShapeClaudeMessages:
		return foldClaude(payload)
	case ShapeGeminiGenerateContent:
		return foldGemini(payload)
	default:
		return nil, fmt.Errorf("toolemu: unknown shape %d", opts.Shape)
	}
}

// ExtractToolChoice returns the effective native tool_choice before FoldRequest strips native tool fields.
func ExtractToolChoice(payload []byte, shape UpstreamShape) ToolChoice {
	switch shape {
	case ShapeOpenAIChat:
		return parseChatToolChoice(gjson.GetBytes(payload, "tool_choice"))
	case ShapeOpenAIResponses:
		return parseResponsesToolChoice(gjson.GetBytes(payload, "tool_choice"))
	case ShapeClaudeMessages:
		return parseClaudeToolChoice(gjson.GetBytes(payload, "tool_choice"))
	case ShapeGeminiGenerateContent:
		choiceNode := gjson.GetBytes(payload, "tool_config.function_calling_config")
		if !choiceNode.Exists() {
			choiceNode = gjson.GetBytes(payload, "toolConfig.functionCallingConfig")
		}
		return parseGeminiToolChoice(choiceNode)
	default:
		return ToolChoiceAuto
	}
}

func foldChat(payload []byte) ([]byte, error) {
	tools, choice, err := extractChatTools(payload)
	if err != nil {
		return nil, err
	}

	// Strip native tool fields unconditionally so the upstream never observes
	// the original protocol when toolemu activates — even on later turns where
	// the client only sends tool_calls / role=tool history without redeclaring
	// tools.
	out, _ := sjson.DeleteBytes(payload, "tools")
	out, _ = sjson.DeleteBytes(out, "tool_choice")

	messages := gjson.GetBytes(out, "messages")
	if !messages.IsArray() {
		return out, nil
	}

	if len(tools) > 0 {
		injection := RenderInjection(tools, choice)

		sysIdx := -1
		messages.ForEach(func(idx, msg gjson.Result) bool {
			if msg.Get("role").String() == "system" {
				sysIdx = int(idx.Int())
				return false
			}
			return true
		})

		if sysIdx >= 0 {
			path := fmt.Sprintf("messages.%d.content", sysIdx)
			existing := gjson.GetBytes(out, path)
			if existing.IsArray() {
				// OpenAI multimodal content: append a new text part instead of
				// collapsing the parts array into a JSON-encoded string.
				var parts []json.RawMessage
				if err := json.Unmarshal([]byte(existing.Raw), &parts); err != nil {
					return nil, fmt.Errorf("toolemu: parse system content parts: %w", err)
				}
				partObj, errPart := marshalSorted(map[string]any{"type": "text", "text": injection})
				if errPart != nil {
					return nil, fmt.Errorf("toolemu: marshal injection part: %w", errPart)
				}
				items := make([]any, 0, len(parts)+1)
				for _, p := range parts {
					items = append(items, p)
				}
				items = append(items, json.RawMessage(partObj))
				merged, errMerge := marshalSorted(items)
				if errMerge != nil {
					return nil, fmt.Errorf("toolemu: marshal merged system content: %w", errMerge)
				}
				out, _ = sjson.SetRawBytes(out, path, merged)
			} else {
				// Plain string content — append injection directly.
				// Use marshalLeafNoEscape so `<tool_protocol>` and friends remain
				// literal in the wire bytes. sjson.SetBytes falls back to
				// encoding/json (with HTML-escape on) for strings containing both a
				// newline and `<`, which would diverge from the insertion path
				// below and fragment the upstream prefix cache.
				raw, errLeaf := marshalLeafNoEscape(existing.String() + "\n" + injection)
				if errLeaf != nil {
					return nil, fmt.Errorf("toolemu: marshal injected system content: %w", errLeaf)
				}
				out, _ = sjson.SetRawBytes(out, path, raw)
			}
		} else {
			// Insert new system message at index 0. Use marshalSorted so the
			// resulting wire bytes carry literal `<tool_call>`/`<tool_protocol>`
			// sentinels instead of HTML-escaped `<...` — this keeps the folded
			// prefix byte-stable with the existing-system branch above (which
			// goes through sjson.SetBytes and does not HTML-escape).
			sys, errSys := marshalSorted(map[string]any{"role": "system", "content": injection})
			if errSys != nil {
				return nil, fmt.Errorf("toolemu: marshal injected system message: %w", errSys)
			}
			var arr []json.RawMessage
			_ = json.Unmarshal([]byte(gjson.GetBytes(out, "messages").Raw), &arr)
			items := make([]any, 0, len(arr)+1)
			items = append(items, json.RawMessage(sys))
			for _, e := range arr {
				items = append(items, json.RawMessage(e))
			}
			merged, errMerge := marshalSorted(items)
			if errMerge != nil {
				return nil, fmt.Errorf("toolemu: marshal merged messages: %w", errMerge)
			}
			out, _ = sjson.SetRawBytes(out, "messages", merged)
		}
	}

	// Always fold history so historical assistant.tool_calls / role=tool
	// messages are converted into <tool_call>/<tool_result> text blocks.
	rawMessages := gjson.GetBytes(out, "messages").Raw
	folded, err := FoldChatHistory([]byte(rawMessages))
	if err != nil {
		return nil, err
	}
	out, _ = sjson.SetRawBytes(out, "messages", folded)
	return out, nil
}

func foldResponses(payload []byte) ([]byte, error) {
	tools, choice, err := extractResponsesTools(payload)
	if err != nil {
		return nil, err
	}

	out, _ := sjson.DeleteBytes(payload, "tools")
	out, _ = sjson.DeleteBytes(out, "tool_choice")

	if len(tools) > 0 {
		injection := RenderInjection(tools, choice)
		existing := gjson.GetBytes(out, "instructions").String()
		combined := existing
		if combined != "" {
			combined += "\n"
		}
		combined += injection
		// Use marshalLeafNoEscape (SetEscapeHTML(false)) so `<tool_protocol>`
		// and related sentinels stay literal — sjson.SetBytes would fall back
		// to encoding/json with HTML-escape on for strings containing `\n<`.
		raw, errLeaf := marshalLeafNoEscape(combined)
		if errLeaf != nil {
			return nil, fmt.Errorf("toolemu: marshal injected instructions: %w", errLeaf)
		}
		out, _ = sjson.SetRawBytes(out, "instructions", raw)
	}

	if input := gjson.GetBytes(out, "input"); input.IsArray() {
		folded, err := FoldResponsesInput([]byte(input.Raw))
		if err != nil {
			return nil, err
		}
		out, _ = sjson.SetRawBytes(out, "input", folded)
	}
	return out, nil
}

func foldClaude(payload []byte) ([]byte, error) {
	tools, choice, err := extractClaudeTools(payload)
	if err != nil {
		return nil, err
	}
	out, _ := sjson.DeleteBytes(payload, "tools")
	out, _ = sjson.DeleteBytes(out, "tool_choice")

	if len(tools) > 0 {
		injection := RenderInjection(tools, choice)
		updated, errInject := appendClaudeSystem(out, injection)
		if errInject != nil {
			return nil, errInject
		}
		out = updated
	}
	if messages := gjson.GetBytes(out, "messages"); messages.IsArray() {
		folded, errFold := FoldClaudeMessages([]byte(messages.Raw))
		if errFold != nil {
			return nil, errFold
		}
		out, _ = sjson.SetRawBytes(out, "messages", folded)
	}
	return out, nil
}

func foldGemini(payload []byte) ([]byte, error) {
	tools, choice, err := extractGeminiTools(payload)
	if err != nil {
		return nil, err
	}
	out, _ := sjson.DeleteBytes(payload, "tools")
	out, _ = sjson.DeleteBytes(out, "tool_config")
	out, _ = sjson.DeleteBytes(out, "toolConfig")

	if len(tools) > 0 {
		injection := RenderInjection(tools, choice)
		part, errPart := marshalSorted(map[string]any{"text": injection})
		if errPart != nil {
			return nil, fmt.Errorf("toolemu: marshal Gemini system injection: %w", errPart)
		}
		out, _ = sjson.SetRawBytes(out, "systemInstruction.parts.-1", part)
		if !gjson.GetBytes(out, "systemInstruction.role").Exists() {
			out, _ = sjson.SetBytes(out, "systemInstruction.role", "system")
		}
	}
	if contents := gjson.GetBytes(out, "contents"); contents.IsArray() {
		folded, errFold := FoldGeminiContents([]byte(contents.Raw))
		if errFold != nil {
			return nil, errFold
		}
		out, _ = sjson.SetRawBytes(out, "contents", folded)
	}
	return out, nil
}

func extractChatTools(payload []byte) ([]ToolSpec, ToolChoice, error) {
	toolsRes := gjson.GetBytes(payload, "tools")
	if !toolsRes.IsArray() {
		return nil, ToolChoiceAuto, nil
	}
	var specs []ToolSpec
	toolsRes.ForEach(func(_, t gjson.Result) bool {
		fn := t.Get("function")
		if !fn.Exists() {
			return true
		}
		schema := json.RawMessage(fn.Get("parameters").Raw)
		if len(schema) == 0 {
			schema = json.RawMessage(`{}`)
		}
		specs = append(specs, ToolSpec{
			Name: fn.Get("name").String(), Description: fn.Get("description").String(),
			SchemaJSON: schema,
		})
		return true
	})
	choice := parseChatToolChoice(gjson.GetBytes(payload, "tool_choice"))
	return specs, choice, nil
}

func extractResponsesTools(payload []byte) ([]ToolSpec, ToolChoice, error) {
	toolsRes := gjson.GetBytes(payload, "tools")
	if !toolsRes.IsArray() {
		return nil, ToolChoiceAuto, nil
	}
	var specs []ToolSpec
	toolsRes.ForEach(func(_, t gjson.Result) bool {
		// Responses tool schema is flatter than chat-completions:
		// {"type":"function","name":"...","description":"...","parameters":{...}}
		if t.Get("type").String() != "function" {
			return true
		}
		schema := json.RawMessage(t.Get("parameters").Raw)
		if len(schema) == 0 {
			schema = json.RawMessage(`{}`)
		}
		specs = append(specs, ToolSpec{
			Name: t.Get("name").String(), Description: t.Get("description").String(),
			SchemaJSON: schema,
		})
		return true
	})
	choice := parseResponsesToolChoice(gjson.GetBytes(payload, "tool_choice"))
	return specs, choice, nil
}

func extractClaudeTools(payload []byte) ([]ToolSpec, ToolChoice, error) {
	toolsRes := gjson.GetBytes(payload, "tools")
	if !toolsRes.IsArray() {
		return nil, ToolChoiceAuto, nil
	}
	var specs []ToolSpec
	toolsRes.ForEach(func(_, t gjson.Result) bool {
		schema := json.RawMessage(t.Get("input_schema").Raw)
		if len(schema) == 0 {
			schema = json.RawMessage(`{}`)
		}
		specs = append(specs, ToolSpec{
			Name: t.Get("name").String(), Description: t.Get("description").String(),
			SchemaJSON: schema,
		})
		return true
	})
	return specs, parseClaudeToolChoice(gjson.GetBytes(payload, "tool_choice")), nil
}

func extractGeminiTools(payload []byte) ([]ToolSpec, ToolChoice, error) {
	toolsRes := gjson.GetBytes(payload, "tools")
	if !toolsRes.IsArray() {
		return nil, ToolChoiceAuto, nil
	}
	var specs []ToolSpec
	toolsRes.ForEach(func(_, tool gjson.Result) bool {
		decls := tool.Get("functionDeclarations")
		if !decls.IsArray() {
			return true
		}
		decls.ForEach(func(_, d gjson.Result) bool {
			schema := json.RawMessage(d.Get("parameters").Raw)
			if len(schema) == 0 {
				schema = json.RawMessage(`{}`)
			}
			specs = append(specs, ToolSpec{
				Name: d.Get("name").String(), Description: d.Get("description").String(),
				SchemaJSON: schema,
			})
			return true
		})
		return true
	})
	choiceNode := gjson.GetBytes(payload, "tool_config.function_calling_config")
	if !choiceNode.Exists() {
		choiceNode = gjson.GetBytes(payload, "toolConfig.functionCallingConfig")
	}
	return specs, parseGeminiToolChoice(choiceNode), nil
}
func parseChatToolChoice(v gjson.Result) ToolChoice {
	if !v.Exists() {
		return ToolChoiceAuto
	}
	if v.Type == gjson.String {
		switch v.String() {
		case "none":
			return ToolChoiceNone
		case "required":
			return ToolChoiceRequired
		}
		return ToolChoiceAuto
	}
	if v.IsObject() {
		if name := v.Get("function.name").String(); name != "" {
			return ToolChoiceNamed(name)
		}
	}
	return ToolChoiceAuto
}

func parseResponsesToolChoice(v gjson.Result) ToolChoice {
	if !v.Exists() {
		return ToolChoiceAuto
	}
	if v.Type == gjson.String {
		switch v.String() {
		case "none":
			return ToolChoiceNone
		case "required":
			return ToolChoiceRequired
		}
		return ToolChoiceAuto
	}
	if v.IsObject() {
		if name := v.Get("name").String(); name != "" {
			return ToolChoiceNamed(name)
		}
	}
	return ToolChoiceAuto
}

func parseClaudeToolChoice(v gjson.Result) ToolChoice {
	if !v.Exists() {
		return ToolChoiceAuto
	}
	switch v.Get("type").String() {
	case "none":
		return ToolChoiceNone
	case "any":
		return ToolChoiceRequired
	case "tool":
		if name := v.Get("name").String(); name != "" {
			return ToolChoiceNamed(name)
		}
	}
	return ToolChoiceAuto
}

func parseGeminiToolChoice(v gjson.Result) ToolChoice {
	if !v.Exists() {
		return ToolChoiceAuto
	}
	switch strings.ToUpper(v.Get("mode").String()) {
	case "NONE":
		return ToolChoiceNone
	case "ANY":
		names := v.Get("allowed_function_names")
		if !names.IsArray() {
			names = v.Get("allowedFunctionNames")
		}
		if names.IsArray() && len(names.Array()) == 1 {
			return ToolChoiceNamed(names.Array()[0].String())
		}
		return ToolChoiceRequired
	}
	return ToolChoiceAuto
}

func appendClaudeSystem(payload []byte, injection string) ([]byte, error) {
	system := gjson.GetBytes(payload, "system")
	if system.IsArray() {
		part, err := marshalSorted(map[string]any{"type": "text", "text": injection})
		if err != nil {
			return nil, fmt.Errorf("toolemu: marshal Claude system injection: %w", err)
		}
		out, _ := sjson.SetRawBytes(payload, "system.-1", part)
		return out, nil
	}
	combined := system.String()
	if combined != "" {
		combined += "\n"
	}
	combined += injection
	raw, err := marshalLeafNoEscape(combined)
	if err != nil {
		return nil, fmt.Errorf("toolemu: marshal Claude system: %w", err)
	}
	out, _ := sjson.SetRawBytes(payload, "system", raw)
	return out, nil
}

func FoldClaudeMessages(messages []byte) ([]byte, error) {
	if !gjson.ValidBytes(messages) || !gjson.ParseBytes(messages).IsArray() {
		return nil, fmt.Errorf("toolemu: Claude messages payload is not a JSON array")
	}
	var raw []json.RawMessage
	if err := json.Unmarshal(messages, &raw); err != nil {
		return nil, err
	}
	out := make([]any, 0, len(raw))
	for _, msg := range raw {
		role := gjson.GetBytes(msg, "role").String()
		content := gjson.GetBytes(msg, "content")
		switch role {
		case "assistant":
			parts, changed := foldClaudeAssistantParts(content)
			if !changed {
				out = append(out, msg)
				continue
			}
			rebuilt, err := rebuildMessageContent(msg, parts)
			if err != nil {
				return nil, err
			}
			out = append(out, json.RawMessage(rebuilt))
		case "user":
			parts, changed := foldClaudeUserParts(content)
			if !changed {
				out = append(out, msg)
				continue
			}
			rebuilt, err := rebuildMessageContent(msg, parts)
			if err != nil {
				return nil, err
			}
			out = append(out, json.RawMessage(rebuilt))
		default:
			out = append(out, msg)
		}
	}
	return marshalSorted(out)
}

func foldClaudeAssistantParts(content gjson.Result) ([]any, bool) {
	if content.Type == gjson.String || !content.IsArray() {
		return nil, false
	}
	parts := make([]any, 0, len(content.Array()))
	changed := false
	content.ForEach(func(_, part gjson.Result) bool {
		if part.Get("type").String() != "tool_use" {
			parts = append(parts, json.RawMessage(part.Raw))
			return true
		}
		obj := map[string]any{
			"id":        part.Get("id").String(),
			"name":      part.Get("name").String(),
			"arguments": json.RawMessage(canonicalJSON(json.RawMessage(part.Get("input").Raw))),
		}
		stable, _ := marshalSorted(obj)
		parts = append(parts, map[string]any{"type": "text", "text": "<tool_call>\n" + string(stable) + "\n</tool_call>"})
		changed = true
		return true
	})
	return parts, changed
}

func foldClaudeUserParts(content gjson.Result) ([]any, bool) {
	if !content.IsArray() {
		return nil, false
	}
	parts := make([]any, 0, len(content.Array()))
	changed := false
	content.ForEach(func(_, part gjson.Result) bool {
		if part.Get("type").String() != "tool_result" {
			parts = append(parts, json.RawMessage(part.Raw))
			return true
		}
		text := fmt.Sprintf("<tool_result tool_call_id=%q>%s</tool_result>", part.Get("tool_use_id").String(), part.Get("content").String())
		parts = append(parts, map[string]any{"type": "text", "text": text})
		changed = true
		return true
	})
	return parts, changed
}

func rebuildMessageContent(orig json.RawMessage, content []any) ([]byte, error) {
	var obj map[string]any
	if err := json.Unmarshal(orig, &obj); err != nil {
		return nil, err
	}
	obj["content"] = content
	return marshalSorted(obj)
}

func FoldGeminiContents(contents []byte) ([]byte, error) {
	if !gjson.ValidBytes(contents) || !gjson.ParseBytes(contents).IsArray() {
		return nil, fmt.Errorf("toolemu: Gemini contents payload is not a JSON array")
	}
	var raw []json.RawMessage
	if err := json.Unmarshal(contents, &raw); err != nil {
		return nil, err
	}
	out := make([]any, 0, len(raw))
	for _, item := range raw {
		parts := gjson.GetBytes(item, "parts")
		partsOut, changed := foldGeminiParts(parts)
		if !changed {
			out = append(out, item)
			continue
		}
		rebuilt, err := rebuildGeminiParts(item, partsOut)
		if err != nil {
			return nil, err
		}
		out = append(out, json.RawMessage(rebuilt))
	}
	return marshalSorted(out)
}

func foldGeminiParts(parts gjson.Result) ([]any, bool) {
	if !parts.IsArray() {
		return nil, false
	}
	out := make([]any, 0, len(parts.Array()))
	changed := false
	parts.ForEach(func(_, part gjson.Result) bool {
		if fc := part.Get("functionCall"); fc.Exists() {
			obj := map[string]any{
				"name":      fc.Get("name").String(),
				"arguments": json.RawMessage(canonicalJSON(json.RawMessage(fc.Get("args").Raw))),
			}
			stable, _ := marshalSorted(obj)
			out = append(out, map[string]any{"text": "<tool_call>\n" + string(stable) + "\n</tool_call>"})
			changed = true
			return true
		}
		if fr := part.Get("functionResponse"); fr.Exists() {
			result := fr.Get("response").Raw
			if result == "" {
				result = "{}"
			}
			text := fmt.Sprintf("<tool_result tool_call_id=%q>%s</tool_result>", fr.Get("name").String(), result)
			out = append(out, map[string]any{"text": text})
			changed = true
			return true
		}
		out = append(out, json.RawMessage(part.Raw))
		return true
	})
	return out, changed
}

func rebuildGeminiParts(orig json.RawMessage, parts []any) ([]byte, error) {
	var obj map[string]any
	if err := json.Unmarshal(orig, &obj); err != nil {
		return nil, err
	}
	obj["parts"] = parts
	return marshalSorted(obj)
}

// HasTools reports whether the upstream payload still carries a non-empty tools array.
func HasTools(payload []byte) bool {
	t := gjson.GetBytes(payload, "tools")
	return t.IsArray() && len(t.Array()) > 0
}

// HasToolArtifacts reports whether the upstream payload carries any native
// tool-calling structures: a non-empty `tools` array, an object-form
// `tool_choice`, historical `assistant.tool_calls` or `role=tool` messages
// (chat-completions), or `function_call` / `function_call_output` items
// (Responses). Used by executors to decide whether toolemu folding is
// necessary — even when the current turn omits the tools declaration but
// the conversation history still references native tool calls.
func HasToolArtifacts(payload []byte) bool {
	if HasTools(payload) {
		return true
	}
	if tc := gjson.GetBytes(payload, "tool_choice"); tc.IsObject() {
		return true
	}
	if messages := gjson.GetBytes(payload, "messages"); messages.IsArray() {
		found := false
		messages.ForEach(func(_, msg gjson.Result) bool {
			if msg.Get("role").String() == "tool" {
				found = true
				return false
			}
			if calls := msg.Get("tool_calls"); calls.IsArray() && len(calls.Array()) > 0 {
				found = true
				return false
			}
			if content := msg.Get("content"); content.IsArray() {
				content.ForEach(func(_, part gjson.Result) bool {
					switch part.Get("type").String() {
					case "tool_use", "tool_result":
						found = true
						return false
					}
					return true
				})
			}
			return !found
		})
		if found {
			return true
		}
	}
	if input := gjson.GetBytes(payload, "input"); input.IsArray() {
		found := false
		input.ForEach(func(_, item gjson.Result) bool {
			switch item.Get("type").String() {
			case "function_call", "function_call_output":
				found = true
				return false
			}
			return true
		})
		if found {
			return true
		}
	}
	if contents := gjson.GetBytes(payload, "contents"); contents.IsArray() {
		found := false
		contents.ForEach(func(_, item gjson.Result) bool {
			if parts := item.Get("parts"); parts.IsArray() {
				parts.ForEach(func(_, part gjson.Result) bool {
					if part.Get("functionCall").Exists() || part.Get("functionResponse").Exists() {
						found = true
						return false
					}
					return true
				})
			}
			return !found
		})
		if found {
			return true
		}
	}
	return false
}
