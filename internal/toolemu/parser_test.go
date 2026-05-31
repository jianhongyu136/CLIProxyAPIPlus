package toolemu

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	log "github.com/sirupsen/logrus"
)

func TestParse_SingleToolCall(t *testing.T) {
	text := `Let me check.
<tool_call>
{"name":"get_weather","arguments":{"city":"SF"}}
</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.ToolCalls) != 1 {
		t.Fatalf("got %d tool_calls, want 1", len(p.ToolCalls))
	}
	if p.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("name = %q", p.ToolCalls[0].Name)
	}
	if !strings.HasPrefix(p.Prose, "Let me check.") {
		t.Fatalf("prose lost: %q", p.Prose)
	}
}

func TestParse_MultipleToolCalls(t *testing.T) {
	text := `<tool_call>{"name":"a","arguments":{}}</tool_call>
<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.ToolCalls) != 2 {
		t.Fatalf("got %d, want 2", len(p.ToolCalls))
	}
	if p.ToolCalls[0].Name != "a" || p.ToolCalls[1].Name != "b" {
		t.Fatalf("order wrong: %v", p.ToolCalls)
	}
}

func TestParse_UnterminatedReturnsError(t *testing.T) {
	_, err := Parse(`<tool_call>{"name":"x"`)
	if err == nil {
		t.Fatal("expected error for unterminated tool_call")
	}
}

func TestParse_MalformedJSONReturnsError(t *testing.T) {
	_, err := Parse(`<tool_call>not json</tool_call>`)
	if err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestParse_LogsRawTextOnFailure(t *testing.T) {
	var buf bytes.Buffer
	oldOut := log.StandardLogger().Out
	oldLevel := log.GetLevel()
	oldFormatter := log.StandardLogger().Formatter
	log.SetOutput(&buf)
	log.SetLevel(log.ErrorLevel)
	log.SetFormatter(&log.JSONFormatter{})
	defer func() {
		log.SetOutput(oldOut)
		log.SetLevel(oldLevel)
		log.SetFormatter(oldFormatter)
	}()

	raw := `<tool_call>{"arguments":{"x":1}}</tool_call>`
	_, err := Parse(raw)
	if err == nil {
		t.Fatal("expected parse failure")
	}
	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("log should be JSON: %v; %q", err, buf.String())
	}
	if entry["msg"] != "tool call parse failed" || entry["raw_text"] != raw {
		t.Fatalf("log should include parse error and raw text, got %#v", entry)
	}
}

func TestParse_StringEncodedArguments(t *testing.T) {
	text := `<tool_call>{"name":"f","arguments":"{\"x\":1}"}</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatal(err)
	}
	if string(p.ToolCalls[0].Arguments) != `{"x":1}` {
		t.Fatalf("expected canonical {\"x\":1}, got %s", p.ToolCalls[0].Arguments)
	}
}

func TestParse_NoToolCallProsePreserved(t *testing.T) {
	p, err := Parse("just plain text")
	if err != nil {
		t.Fatal(err)
	}
	if p.Prose != "just plain text" || len(p.ToolCalls) != 0 {
		t.Fatalf("unexpected parse: %+v", p)
	}
}

func TestParse_IgnoresToolCallInsideCodeFence(t *testing.T) {
	text := "```\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\n```\n"
	p, err := Parse(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.ToolCalls) != 0 {
		t.Fatalf("expected no tool calls, got %+v", p.ToolCalls)
	}
	if !strings.Contains(p.Prose, "<tool_call>") {
		t.Fatalf("fence content should be prose: %q", p.Prose)
	}
}

func TestParse_IgnoresToolCallInsideInlineCode(t *testing.T) {
	text := "Use `<tool_call>` to call tools"
	p, err := Parse(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.ToolCalls) != 0 {
		t.Fatalf("expected no tool calls, got %+v", p.ToolCalls)
	}
	if p.Prose != text {
		t.Fatalf("got prose %q", p.Prose)
	}
}

func TestParse_CaseInsensitiveSentinels(t *testing.T) {
	for _, tc := range []struct {
		name string
		text string
	}{
		{"upper", `<TOOL_CALL>{"name":"x","arguments":{}}</TOOL_CALL>`},
		{"mixed-open", `<Tool_Call>{"name":"x","arguments":{}}</tool_call>`},
		{"mixed-close", `<tool_call>{"name":"x","arguments":{}}</Tool_Call>`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := Parse(tc.text)
			if err != nil {
				t.Fatal(err)
			}
			if len(p.ToolCalls) != 1 || p.ToolCalls[0].Name != "x" {
				t.Fatalf("expected single call to x, got %+v", p)
			}
		})
	}
}

func TestParse_TolerantArguments(t *testing.T) {
	for _, tc := range []struct {
		name     string
		text     string
		wantArgs string
	}{
		{"missing", `<tool_call>{"name":"x"}</tool_call>`, `{}`},
		{"null", `<tool_call>{"name":"x","arguments":null}</tool_call>`, `{}`},
		{"empty-string", `<tool_call>{"name":"x","arguments":""}</tool_call>`, `{}`},
		{"whitespace-string", `<tool_call>{"name":"x","arguments":"   "}</tool_call>`, `{}`},
		{"trailing-comma-object", `<tool_call>{"name":"x","arguments":{"a":1,}}</tool_call>`, `{"a":1}`},
		{"trailing-comma-nested", `<tool_call>{"name":"x","arguments":{"a":[1,2,],"b":{"k":1,}}}</tool_call>`, `{"a":[1,2],"b":{"k":1}}`},
		{"string-encoded-trailing-comma", `<tool_call>{"name":"x","arguments":"{\"a\":1,}"}</tool_call>`, `{"a":1}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := Parse(tc.text)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			if len(p.ToolCalls) != 1 {
				t.Fatalf("expected 1 call, got %d", len(p.ToolCalls))
			}
			if got := string(p.ToolCalls[0].Arguments); got != tc.wantArgs {
				t.Fatalf("arguments = %s, want %s", got, tc.wantArgs)
			}
		})
	}
}

func TestParse_StripsNamespacePrefix(t *testing.T) {
	for _, tc := range []struct {
		input string
		want  string
	}{
		{"functions.get_weather", "get_weather"},
		{"tools.get_weather", "get_weather"},
		{"FUNCTIONS.Get_Weather", "Get_Weather"},
		{"function.get_weather", "get_weather"},
		{"tool.get_weather", "get_weather"},
		{"get_weather", "get_weather"},
		{"vendor.get_weather", "vendor.get_weather"}, // unknown ns preserved
	} {
		t.Run(tc.input, func(t *testing.T) {
			text := `<tool_call>{"name":"` + tc.input + `","arguments":{}}</tool_call>`
			p, err := Parse(text)
			if err != nil {
				t.Fatal(err)
			}
			if p.ToolCalls[0].Name != tc.want {
				t.Fatalf("name = %q, want %q", p.ToolCalls[0].Name, tc.want)
			}
		})
	}
}

// TestParse_EmptyNameFails guards against over-eager normalization: a bare
// "functions." would normalize to "" which is not a usable tool name.
func TestParse_EmptyNameFails(t *testing.T) {
	for _, text := range []string{
		`<tool_call>{"name":"","arguments":{}}</tool_call>`,
		`<tool_call>{"name":"functions.","arguments":{}}</tool_call>`,
		`<tool_call>{"name":"   ","arguments":{}}</tool_call>`,
	} {
		if _, err := Parse(text); err == nil {
			t.Fatalf("expected error for empty/whitespace name, got nil; input=%q", text)
		}
	}
}

func TestParse_MalformedJSONStillFails(t *testing.T) {
	// Trailing-comma cleanup must not rescue genuinely broken payloads.
	if _, err := Parse(`<tool_call>{"name":"x","arguments":{a:1}}</tool_call>`); err == nil {
		t.Fatal("expected error for unquoted key")
	}
	if _, err := Parse(`<tool_call>{"name":"x","arguments":"not an object"}</tool_call>`); err == nil {
		t.Fatal("expected error for non-object string arguments")
	}
}

// TestParse_ClosingTagInsideStringValue is the headline brace-tracking test:
// when the model writes a literal `</tool_call>` substring inside a JSON string
// argument (common when a code-generation tool's `content` parameter contains
// template-like text), the parser must NOT mis-split on that embedded
// sentinel. Brace tracking respects JSON string literals so depth-aware
// scanning finds the true closing `}` first.
func TestParse_ClosingTagInsideStringValue(t *testing.T) {
	text := `<tool_call>{"name":"write_file","arguments":{"path":"x.md","content":"snippet contains </tool_call> literally"}}</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(p.ToolCalls) != 1 {
		t.Fatalf("expected 1 call, got %d (prose=%q)", len(p.ToolCalls), p.Prose)
	}
	if p.ToolCalls[0].Name != "write_file" {
		t.Fatalf("name = %q", p.ToolCalls[0].Name)
	}
	// The embedded sentinel must survive intact in the canonicalized arguments.
	if !strings.Contains(string(p.ToolCalls[0].Arguments), "snippet contains </tool_call> literally") {
		t.Fatalf("embedded sentinel lost: %s", p.ToolCalls[0].Arguments)
	}
}

// TestParse_OpenTagInsideStringValue covers the symmetric case: a `<tool_call>`
// substring appearing inside a JSON string of an earlier valid call must not
// be treated as the start of a second block.
func TestParse_OpenTagInsideStringValue(t *testing.T) {
	text := `<tool_call>{"name":"write_file","arguments":{"content":"see <tool_call> example"}}</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(p.ToolCalls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(p.ToolCalls))
	}
}

func TestParse_NestedBracesInArguments(t *testing.T) {
	text := `<tool_call>{"name":"x","arguments":{"a":{"b":[1,2,{"c":3}]},"d":{}}}</tool_call>`
	p, err := Parse(text)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(p.ToolCalls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(p.ToolCalls))
	}
	if got := string(p.ToolCalls[0].Arguments); got != `{"a":{"b":[1,2,{"c":3}]},"d":{}}` {
		t.Fatalf("canonical arguments mismatch: %s", got)
	}
}

// TestParse_MissingClosingTagAcceptedWhenJSONComplete relaxes the strict
// sentinel pairing: when the JSON body is balanced, the closing </tool_call>
// tag becomes optional. Models that forget the closing tag should still
// produce a usable tool call rather than triggering a retry.
func TestParse_MissingClosingTagAcceptedWhenJSONComplete(t *testing.T) {
	text := `<tool_call>{"name":"x","arguments":{"k":1}} trailing prose`
	p, err := Parse(text)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if len(p.ToolCalls) != 1 || p.ToolCalls[0].Name != "x" {
		t.Fatalf("expected call to x, got %+v", p)
	}
	if !strings.Contains(p.Prose, "trailing prose") {
		t.Fatalf("trailing prose lost: %q", p.Prose)
	}
}

// TestParse_HuJSONComments verifies the hujson fallback layer: line and block
// comments inside the JSON body are stripped before unmarshal. Models that
// annotate their arguments (`{"x":1 // explanation}`) should not hit retry.
func TestParse_HuJSONComments(t *testing.T) {
	for _, tc := range []struct {
		name string
		text string
	}{
		{"line-comment", `<tool_call>{
  "name": "x",
  "arguments": { "k": 1 } // trailing comment
}</tool_call>`},
		{"block-comment", `<tool_call>{
  "name": "x",
  /* block
     comment */
  "arguments": { "k": 1 }
}</tool_call>`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := Parse(tc.text)
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			if len(p.ToolCalls) != 1 || p.ToolCalls[0].Name != "x" {
				t.Fatalf("expected call to x, got %+v", p)
			}
			if got := string(p.ToolCalls[0].Arguments); got != `{"k":1}` {
				t.Fatalf("arguments = %s, want {\"k\":1}", got)
			}
		})
	}
}
