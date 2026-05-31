package toolemu

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/tidwall/gjson"
)

func contains(s, sub string) bool { return bytes.Contains([]byte(s), []byte(sub)) }
func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func TestFoldRequest_ChatStripsToolsAndAppendsToSystem(t *testing.T) {
	payload := []byte(`{
		"model": "m",
		"messages": [
			{"role":"system","content":"You are X."},
			{"role":"user","content":"hi"}
		],
		"tools": [
			{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools must be removed:\n%s", out)
	}
	sys := gjson.GetBytes(out, "messages.0.content").String()
	if !contains(sys, "<tools_doc>") || !contains(sys, "<tool_protocol>") {
		t.Fatalf("system content missing injection:\n%s", sys)
	}
	if !startsWith(sys, "You are X.") {
		t.Fatalf("original system content must remain at the start:\n%s", sys)
	}
}

func TestFoldRequest_ChatCreatesSystemWhenAbsent(t *testing.T) {
	payload := []byte(`{
		"model": "m",
		"messages": [{"role":"user","content":"hi"}],
		"tools": [{"type":"function","function":{"name":"f","description":"","parameters":{}}}]
	}`)
	out, _ := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	firstRole := gjson.GetBytes(out, "messages.0.role").String()
	if firstRole != "system" {
		t.Fatalf("expected system message at index 0, got %q\n%s", firstRole, out)
	}
}

func TestFoldRequest_ChatToolChoiceClause(t *testing.T) {
	payload := []byte(`{
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}],
		"tool_choice":"required"
	}`)
	out, _ := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if !bytes.Contains(out, []byte("You MUST call at least one tool")) {
		t.Fatalf("required clause missing:\n%s", out)
	}
	if gjson.GetBytes(out, "tool_choice").Exists() {
		t.Fatalf("tool_choice must be stripped:\n%s", out)
	}
}

func TestFoldRequest_ByteIdenticalForSameInput(t *testing.T) {
	payload := []byte(`{"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}]}`)
	a, _ := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	b, _ := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if !bytes.Equal(a, b) {
		t.Fatalf("FoldRequest must be deterministic\nA: %s\nB: %s", a, b)
	}
}

func TestFoldRequest_ResponsesStripsToolsAndAppendsInstructions(t *testing.T) {
	payload := []byte(`{
		"model":"m",
		"instructions":"You are X.",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
		"tools":[{"type":"function","name":"f","description":"d","parameters":{"type":"object"}}]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIResponses})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools must be stripped:\n%s", out)
	}
	instr := gjson.GetBytes(out, "instructions").String()
	if !startsWith(instr, "You are X.") {
		t.Fatalf("original instructions must remain at front:\n%s", instr)
	}
	if !contains(instr, "<tool_protocol>") {
		t.Fatalf("missing protocol block:\n%s", instr)
	}
}

func TestFoldRequest_ChatFoldsHistoryWhenToolsAbsent(t *testing.T) {
	// Multi-turn continuation where the client carries assistant.tool_calls
	// and role=tool history but no longer redeclares `tools`. toolemu must
	// still rewrite these artifacts into <tool_call>/<tool_result> text so
	// the upstream never observes the native tool-calling protocol.
	payload := []byte(`{
		"model":"m",
		"messages":[
			{"role":"user","content":"call the weather tool"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"sf\"}"}}]},
			{"role":"tool","tool_call_id":"c1","content":"sunny"}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools should remain absent:\n%s", out)
	}
	// The assistant message must have its tool_calls folded into a text block.
	asst := gjson.GetBytes(out, "messages.1")
	if asst.Get("tool_calls").Exists() {
		t.Fatalf("assistant.tool_calls must be removed:\n%s", asst.Raw)
	}
	if !contains(asst.Get("content").String(), "<tool_call>") {
		t.Fatalf("assistant content missing folded tool_call block:\n%s", asst.Raw)
	}
	// The tool message must have been converted to a user message carrying
	// a <tool_result> block.
	follow := gjson.GetBytes(out, "messages.2")
	if follow.Get("role").String() != "user" {
		t.Fatalf("role=tool must be folded into a user message, got %q\n%s", follow.Get("role").String(), follow.Raw)
	}
	if !contains(follow.Get("content").String(), "<tool_result") {
		t.Fatalf("folded content missing tool_result block:\n%s", follow.Raw)
	}
	// No <tools_doc>/<tool_protocol> injection should occur when tools are
	// absent — the historical artifacts are folded but the request carries
	// no current tool declarations.
	if bytes.Contains(out, []byte("<tools_doc>")) {
		t.Fatalf("tools_doc should not be injected without current tools:\n%s", out)
	}
}

func TestFoldRequest_ResponsesFoldsHistoryWhenToolsAbsent(t *testing.T) {
	payload := []byte(`{
		"model":"m",
		"instructions":"",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
			{"type":"function_call","id":"c1","name":"get_weather","arguments":"{\"loc\":\"sf\"}"},
			{"type":"function_call_output","call_id":"c1","output":"sunny"}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIResponses})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools should remain absent:\n%s", out)
	}
	if bytes.Contains(out, []byte("<tool_protocol>")) {
		t.Fatalf("tool_protocol should not be injected without current tools:\n%s", out)
	}
	// All function_call / function_call_output items must be folded out of input.
	gjson.GetBytes(out, "input").ForEach(func(_, item gjson.Result) bool {
		switch item.Get("type").String() {
		case "function_call", "function_call_output":
			t.Fatalf("native call item must be folded:\n%s", item.Raw)
		}
		return true
	})
	if !bytes.Contains(out, []byte("<tool_call>")) {
		t.Fatalf("output missing <tool_call> text block:\n%s", out)
	}
	if !bytes.Contains(out, []byte("<tool_result")) {
		t.Fatalf("output missing <tool_result> text block:\n%s", out)
	}
}

func TestFoldRequest_ChatFoldsHistoryWithoutVolatileToolIDs(t *testing.T) {
	payload := []byte(`{
		"model":"m",
		"messages":[
			{"role":"user","content":"call the weather tool"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_ephemeral_123","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"sf\"}"}}]},
			{"role":"tool","tool_call_id":"call_ephemeral_123","content":"sunny"}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(out, []byte("call_ephemeral_123")) {
		t.Fatalf("folded prompt must not contain volatile tool call id:\n%s", out)
	}
	assistant := gjson.GetBytes(out, "messages.1.content").String()
	result := gjson.GetBytes(out, "messages.2.content").String()
	if !contains(assistant, `"index":0`) || result != `<tool_result index="0">sunny</tool_result>` {
		t.Fatalf("tool result should still match its call with stable index:\n%s", out)
	}
}

func TestFoldRequest_ResponsesFoldsHistoryWithoutVolatileToolIDs(t *testing.T) {
	payload := []byte(`{
		"model":"m",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
			{"type":"function_call","id":"call_ephemeral_123","call_id":"call_ephemeral_123","name":"get_weather","arguments":"{\"loc\":\"sf\"}"},
			{"type":"function_call_output","call_id":"call_ephemeral_123","output":"sunny"}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIResponses})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(out, []byte("call_ephemeral_123")) {
		t.Fatalf("folded prompt must not contain volatile tool call id:\n%s", out)
	}
	callText := gjson.GetBytes(out, "input.1.content.0.text").String()
	resultText := gjson.GetBytes(out, "input.2.content.0.text").String()
	if !contains(callText, `"index":0`) || resultText != `<tool_result index="0">sunny</tool_result>` {
		t.Fatalf("tool result should still be folded with a stable index:\n%s", out)
	}
}

func TestFoldRequest_ChatFoldsMultipleToolResultsWithStableIndexes(t *testing.T) {
	payload := []byte(`{
		"model":"m",
		"messages":[
			{"role":"user","content":"call tools"},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_random_a","type":"function","function":{"name":"first","arguments":"{}"}},
				{"id":"call_random_b","type":"function","function":{"name":"second","arguments":"{}"}}
			]},
			{"role":"tool","tool_call_id":"call_random_a","content":"one"},
			{"role":"tool","tool_call_id":"call_random_b","content":"two"}
		]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(out, []byte("call_random_")) {
		t.Fatalf("folded prompt must not contain volatile tool call ids:\n%s", out)
	}
	assistant := gjson.GetBytes(out, "messages.1.content").String()
	result := gjson.GetBytes(out, "messages.2.content").String()
	for _, want := range []string{
		`"index":0`,
		`"index":1`,
		`<tool_result index="0">one</tool_result>`,
		`<tool_result index="1">two</tool_result>`,
	} {
		if !contains(assistant+result, want) {
			t.Fatalf("missing %s in:\n%s", want, out)
		}
	}
}

func TestHasToolArtifacts(t *testing.T) {
	cases := []struct {
		name    string
		payload []byte
		want    bool
	}{
		{"empty", []byte(`{}`), false},
		{"plain chat", []byte(`{"messages":[{"role":"user","content":"hi"}]}`), false},
		{"tools array", []byte(`{"tools":[{"type":"function","function":{"name":"f"}}]}`), true},
		{"object tool_choice", []byte(`{"tool_choice":{"type":"function","function":{"name":"f"}}}`), true},
		{"string tool_choice", []byte(`{"tool_choice":"auto"}`), false},
		{"chat tool_calls history", []byte(`{"messages":[{"role":"assistant","tool_calls":[{"id":"c"}]}]}`), true},
		{"chat tool role", []byte(`{"messages":[{"role":"tool","tool_call_id":"c","content":"r"}]}`), true},
		{"responses function_call", []byte(`{"input":[{"type":"function_call","name":"f"}]}`), true},
		{"responses function_call_output", []byte(`{"input":[{"type":"function_call_output","output":"r"}]}`), true},
		{"responses plain", []byte(`{"input":[{"type":"message","role":"user"}]}`), false},
		{"claude tool_use history", []byte(`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":{}}]}]}`), true},
		{"claude tool_result history", []byte(`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}]}]}`), true},
		{"gemini functionCall history", []byte(`{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f","args":{}}}]}]}`), true},
		{"gemini functionResponse history", []byte(`{"contents":[{"role":"user","parts":[{"functionResponse":{"name":"f","response":{}}}]}]}`), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := HasToolArtifacts(c.payload); got != c.want {
				t.Fatalf("HasToolArtifacts(%s) = %v, want %v", c.name, got, c.want)
			}
		})
	}
}

func TestFoldRequest_ChatAppendsToArraySystemContent(t *testing.T) {
	// Anthropic→OpenAI translation may produce system.content as an array of
	// text parts (multimodal-style). toolemu must append the injection as a
	// new text part instead of collapsing the parts into a JSON-encoded string.
	payload := []byte(`{
		"model":"m",
		"messages":[
			{"role":"system","content":[
				{"type":"text","text":"You are X."},
				{"type":"text","text":"More guidance."}
			]},
			{"role":"user","content":"hi"}
		],
		"tools":[{"type":"function","function":{"name":"f","description":"","parameters":{}}}]
	}`)
	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeOpenAIChat})
	if err != nil {
		t.Fatal(err)
	}
	sys := gjson.GetBytes(out, "messages.0.content")
	if !sys.IsArray() {
		t.Fatalf("system.content must stay as an array, got: %s", sys.Raw)
	}
	parts := sys.Array()
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts (2 original + 1 injection), got %d: %s", len(parts), sys.Raw)
	}
	if parts[0].Get("text").String() != "You are X." {
		t.Fatalf("first part must preserve original text, got: %s", parts[0].Raw)
	}
	last := parts[2]
	if last.Get("type").String() != "text" {
		t.Fatalf("injection part must be type=text, got: %s", last.Raw)
	}
	injectedText := last.Get("text").String()
	if !contains(injectedText, "<tools_doc>") || !contains(injectedText, "<tool_protocol>") {
		t.Fatalf("injection part missing prompt blocks: %s", injectedText)
	}
}

func TestFoldRequest_ClaudeFoldsToolsAndHistory(t *testing.T) {
	payload := []byte(`{
		"model":"claude-test",
		"system":"You are X.",
		"messages":[
			{"role":"user","content":"weather"},
			{"role":"assistant","content":[{"type":"text","text":"checking"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"loc":"sf"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}]}
		],
		"tools":[{"name":"get_weather","description":"weather","input_schema":{"type":"object","properties":{"loc":{"type":"string"}}}}],
		"tool_choice":{"type":"tool","name":"get_weather"}
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeClaudeMessages})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools must be stripped:\n%s", out)
	}
	if gjson.GetBytes(out, "tool_choice").Exists() {
		t.Fatalf("tool_choice must be stripped:\n%s", out)
	}
	system := gjson.GetBytes(out, "system").String()
	if !startsWith(system, "You are X.") || !contains(system, "<tools_doc>") || !contains(system, "<tool_protocol>") {
		t.Fatalf("system missing injection:\n%s", system)
	}
	asst := gjson.GetBytes(out, "messages.1.content")
	if !asst.IsArray() || asst.Array()[0].Get("type").String() != "text" {
		t.Fatalf("assistant content must stay a text-parts array: %s", asst.Raw)
	}
	text := asst.Array()[0].Get("text").String() + asst.Array()[1].Get("text").String()
	if !contains(text, "checking") || !contains(text, "<tool_call>") || !contains(text, `"name":"get_weather"`) {
		t.Fatalf("assistant missing folded tool_call: %s", asst.Raw)
	}
	user := gjson.GetBytes(out, "messages.2.content")
	if !user.IsArray() || !contains(user.Array()[0].Get("text").String(), "<tool_result") {
		t.Fatalf("tool_result must be folded into user text: %s", user.Raw)
	}
}

func TestFoldRequest_GeminiFoldsToolsAndHistory(t *testing.T) {
	payload := []byte(`{
		"model":"gemini-test",
		"systemInstruction":{"role":"system","parts":[{"text":"You are X."}]},
		"contents":[
			{"role":"user","parts":[{"text":"weather"}]},
			{"role":"model","parts":[{"text":"checking"},{"functionCall":{"name":"get_weather","args":{"loc":"sf"}}}]},
			{"role":"user","parts":[{"functionResponse":{"name":"get_weather","response":{"result":"sunny"}}}]}
		],
		"tools":[{"functionDeclarations":[{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"loc":{"type":"string"}}}}]}],
		"tool_config":{"function_calling_config":{"mode":"ANY","allowed_function_names":["get_weather"]}}
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeGeminiGenerateContent})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "tools").Exists() {
		t.Fatalf("tools must be stripped:\n%s", out)
	}
	if gjson.GetBytes(out, "tool_config").Exists() {
		t.Fatalf("tool_config must be stripped:\n%s", out)
	}
	sysParts := gjson.GetBytes(out, "systemInstruction.parts")
	if got := sysParts.Array()[len(sysParts.Array())-1].Get("text").String(); !contains(got, "<tools_doc>") || !contains(got, "<tool_protocol>") {
		t.Fatalf("systemInstruction missing injection: %s", sysParts.Raw)
	}
	modelParts := gjson.GetBytes(out, "contents.1.parts")
	if !modelParts.IsArray() || modelParts.Array()[0].Get("text").String() == "" {
		t.Fatalf("model functionCall must preserve text and fold into text part: %s", modelParts.Raw)
	}
	if !contains(modelParts.Raw, "<tool_call>") {
		t.Fatalf("model text missing folded tool_call: %s", modelParts.Raw)
	}
	userParts := gjson.GetBytes(out, "contents.2.parts")
	if !contains(userParts.Array()[0].Get("text").String(), "<tool_result") {
		t.Fatalf("functionResponse must fold into text part: %s", userParts.Raw)
	}
}

func TestFoldRequest_GeminiCamelCaseToolConfigNone(t *testing.T) {
	payload := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"tools":[{"functionDeclarations":[{"name":"get_weather","description":"weather","parameters":{"type":"object"}}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"NONE"}}
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeGeminiGenerateContent})
	if err != nil {
		t.Fatal(err)
	}
	if gjson.GetBytes(out, "toolConfig").Exists() || gjson.GetBytes(out, "tool_config").Exists() {
		t.Fatalf("native tool config must be stripped:\n%s", out)
	}
	sysText := gjson.GetBytes(out, "systemInstruction.parts.0.text").String()
	if !contains(sysText, "You MUST NOT call any tool") {
		t.Fatalf("NONE clause missing:\n%s", sysText)
	}
}

func TestFoldRequest_GeminiCamelCaseAllowedFunctionNames(t *testing.T) {
	payload := []byte(`{
		"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"tools":[{"functionDeclarations":[{"name":"get_weather","description":"weather","parameters":{"type":"object"}}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["get_weather"]}}
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeGeminiGenerateContent})
	if err != nil {
		t.Fatal(err)
	}
	sysText := gjson.GetBytes(out, "systemInstruction.parts.0.text").String()
	if !contains(sysText, `You MUST call the tool named "get_weather"`) {
		t.Fatalf("named clause missing:\n%s", sysText)
	}
}

func TestFoldRequest_ClaudePreservesNonToolParts(t *testing.T) {
	payload := []byte(`{
		"messages":[
			{"role":"assistant","content":[
				{"type":"text","text":"checking"},
				{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"loc":"sf"}},
				{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}
			]},
			{"role":"user","content":[
				{"type":"text","text":"result follows"},
				{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
				{"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"}}
			]}
		]
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeClaudeMessages})
	if err != nil {
		t.Fatal(err)
	}
	asst := gjson.GetBytes(out, "messages.0.content")
	if !contains(asst.Raw, "<tool_call>") || !contains(asst.Raw, `"type":"image"`) {
		t.Fatalf("assistant should preserve image and fold tool_use:\n%s", asst.Raw)
	}
	user := gjson.GetBytes(out, "messages.1.content")
	if !contains(user.Raw, "result follows") || !contains(user.Raw, "<tool_result") || !contains(user.Raw, `"type":"document"`) {
		t.Fatalf("user should preserve text/document and fold tool_result:\n%s", user.Raw)
	}
}

func TestFoldRequest_GeminiPreservesNonToolParts(t *testing.T) {
	payload := []byte(`{
		"contents":[{"role":"model","parts":[
			{"text":"checking"},
			{"inlineData":{"mimeType":"image/png","data":"abc"}},
			{"functionCall":{"name":"get_weather","args":{"loc":"sf"}}}
		]}]
	}`)

	out, err := FoldRequest(payload, FoldOpts{Shape: ShapeGeminiGenerateContent})
	if err != nil {
		t.Fatal(err)
	}
	parts := gjson.GetBytes(out, "contents.0.parts")
	if !contains(parts.Raw, `"inlineData"`) || !contains(parts.Raw, "<tool_call>") || !contains(parts.Raw, "checking") {
		t.Fatalf("Gemini parts should preserve inlineData and fold functionCall:\n%s", parts.Raw)
	}
}

func TestBuildClaudeMessage_ProseAndToolUse(t *testing.T) {
	out, err := BuildClaudeMessage(Parsed{
		Prose:     "checking",
		ToolCalls: []ParsedToolCall{{Name: "get_weather", Arguments: []byte(`{"loc":"sf"}`)}},
	}, UpstreamMeta{Provider: "p", Model: "m", ResponseID: "r1"})
	if err != nil {
		t.Fatal(err)
	}
	if got := gjson.GetBytes(out, "type").String(); got != "message" {
		t.Fatalf("type=%q: %s", got, out)
	}
	if got := gjson.GetBytes(out, "stop_reason").String(); got != "tool_use" {
		t.Fatalf("stop_reason=%q: %s", got, out)
	}
	if got := gjson.GetBytes(out, "content.0.text").String(); got != "checking" {
		t.Fatalf("text=%q: %s", got, out)
	}
	tool := gjson.GetBytes(out, "content.1")
	if tool.Get("type").String() != "tool_use" || tool.Get("name").String() != "get_weather" {
		t.Fatalf("unexpected tool block: %s", tool.Raw)
	}
	if !startsWith(tool.Get("id").String(), "toolu_") {
		t.Fatalf("Claude tool id must use toolu_ prefix: %s", tool.Raw)
	}
	if got := tool.Get("input.loc").String(); got != "sf" {
		t.Fatalf("input.loc=%q: %s", got, tool.Raw)
	}
}

func TestBuildGeminiGenerateContent_ProseAndFunctionCall(t *testing.T) {
	out, err := BuildGeminiGenerateContent(Parsed{
		Prose:     "checking",
		ToolCalls: []ParsedToolCall{{Name: "get_weather", Arguments: []byte(`{"loc":"sf"}`)}},
	}, UpstreamMeta{Model: "gemini-test"})
	if err != nil {
		t.Fatal(err)
	}
	if got := gjson.GetBytes(out, "candidates.0.content.role").String(); got != "model" {
		t.Fatalf("role=%q: %s", got, out)
	}
	if got := gjson.GetBytes(out, "candidates.0.content.parts.0.text").String(); got != "checking" {
		t.Fatalf("text=%q: %s", got, out)
	}
	call := gjson.GetBytes(out, "candidates.0.content.parts.1.functionCall")
	if call.Get("name").String() != "get_weather" || call.Get("args.loc").String() != "sf" {
		t.Fatalf("unexpected functionCall: %s", call.Raw)
	}
	if got := gjson.GetBytes(out, "candidates.0.finishReason").String(); got != "STOP" {
		t.Fatalf("finishReason=%q: %s", got, out)
	}
}

func TestParseAndRetry_M2ShapesAppendRetryInstruction(t *testing.T) {
	cases := []struct {
		name       string
		shape      UpstreamShape
		payload    []byte
		body       []byte
		retryProbe string
	}{
		{
			name:       "claude",
			shape:      ShapeClaudeMessages,
			payload:    []byte(`{"messages":[{"role":"user","content":"hi"}]}`),
			body:       []byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"` + malformedToolCall + `"}]}`),
			retryProbe: "messages.1.content.0.text",
		},
		{
			name:       "gemini",
			shape:      ShapeGeminiGenerateContent,
			payload:    []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
			body:       []byte(`{"candidates":[{"content":{"parts":[{"text":"` + malformedToolCall + `"}]}}]}`),
			retryProbe: "contents.1.parts.0.text",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var captured [][]byte
			send := func(_ context.Context, p []byte) ([]byte, error) {
				captured = append(captured, append([]byte(nil), p...))
				if len(captured) == 1 {
					return c.body, nil
				}
				return []byte(`{"id":"ok","model":"m","content":[{"type":"text","text":"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"}],"candidates":[{"content":{"parts":[{"text":"<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>"}]}}]}`), nil
			}
			_, err := ParseAndRetry(context.Background(), c.payload, send, c.shape, RetryPolicy{Attempts: 1}, ToolChoiceAuto)
			if err != nil {
				t.Fatal(err)
			}
			if len(captured) != 2 {
				t.Fatalf("want 2 sends, got %d", len(captured))
			}
			if !contains(gjson.GetBytes(captured[1], c.retryProbe).String(), "previous response did not contain") {
				t.Fatalf("retry instruction missing in %s: %s", c.retryProbe, captured[1])
			}
		})
	}
}

func TestExtractAssistantText_ClaudeAndGemini(t *testing.T) {
	claudeBody := []byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}],"usage":{"input_tokens":1}}`)
	text, meta, err := ExtractAssistantText(claudeBody, ShapeClaudeMessages)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello world" || meta.ResponseID != "msg_1" || meta.Model != "claude-test" || !json.Valid(meta.UsagePayload) {
		t.Fatalf("unexpected Claude extraction text=%q meta=%+v", text, meta)
	}

	geminiBody := []byte(`{"modelVersion":"gemini-test","candidates":[{"content":{"parts":[{"text":"hello"},{"text":" world"}]}}],"usageMetadata":{"promptTokenCount":1}}`)
	text, meta, err = ExtractAssistantText(geminiBody, ShapeGeminiGenerateContent)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello world" || meta.Model != "gemini-test" || !json.Valid(meta.UsagePayload) {
		t.Fatalf("unexpected Gemini extraction text=%q meta=%+v", text, meta)
	}
}
