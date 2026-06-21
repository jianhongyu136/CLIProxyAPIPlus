package toolemu

import (
	"strings"
	"testing"
)

type spyEvents struct {
	proseDeltas     []string
	reasoningDeltas []string
	toolStarts      []struct {
		index    int
		id, name string
	}
	argsDeltas []struct {
		index int
		delta string
	}
	toolEnds  []int
	completed bool
	err       error
}

func (s *spyEvents) events() StreamEvents {
	return StreamEvents{
		OnProseDelta:     func(d string) { s.proseDeltas = append(s.proseDeltas, d) },
		OnReasoningDelta: func(d string) { s.reasoningDeltas = append(s.reasoningDeltas, d) },
		OnToolCallStart: func(index int, id, name string) {
			s.toolStarts = append(s.toolStarts, struct {
				index    int
				id, name string
			}{index, id, name})
		},
		OnArgsDelta: func(index int, delta string) {
			s.argsDeltas = append(s.argsDeltas, struct {
				index int
				delta string
			}{index, delta})
		},
		OnToolCallEnd: func(index int) { s.toolEnds = append(s.toolEnds, index) },
		OnComplete:    func() { s.completed = true },
		OnError:       func(err error) { s.err = err },
	}
}

func (s *spyEvents) allProse() string {
	return strings.Join(s.proseDeltas, "")
}

func (s *spyEvents) allReasoning() string {
	return strings.Join(s.reasoningDeltas, "")
}

func (s *spyEvents) allArgs(index int) string {
	var sb strings.Builder
	for _, a := range s.argsDeltas {
		if a.index == index {
			sb.WriteString(a.delta)
		}
	}
	return sb.String()
}

func TestStreamParser_ProseOnly(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed("Hello world")
	p.Close()
	if spy.allProse() != "Hello world" {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if len(spy.toolStarts) != 0 {
		t.Fatal("unexpected tool starts")
	}
	if !spy.completed {
		t.Fatal("OnComplete not called")
	}
}

func TestStreamParser_SingleToolCall(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`Hello <tool_call>{"name":"get_weather","arguments":{"city":"NYC"}}</tool_call> done`)
	p.Close()
	if spy.allProse() != "Hello  done" {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "get_weather" {
		t.Fatalf("tool starts: %+v", spy.toolStarts)
	}
	if spy.allArgs(0) != `{"city":"NYC"}` {
		t.Fatalf("got args %q", spy.allArgs(0))
	}
	if len(spy.toolEnds) != 1 {
		t.Fatal("missing tool end")
	}
}

func TestStreamParser_FragmentedTag(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	fragments := []string{"He", "llo <to", "ol_c", "all>", `{"na`, `me":"f","ar`, `guments":{"x":1}}`, "</tool_call>"}
	for _, f := range fragments {
		p.Feed(f)
	}
	p.Close()
	if spy.allProse() != "Hello " {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "f" {
		t.Fatalf("tool starts: %+v", spy.toolStarts)
	}
	if spy.allArgs(0) != `{"x":1}` {
		t.Fatalf("got args %q", spy.allArgs(0))
	}
}

func TestStreamParser_SkipsWhitespaceBeforeCloseTag(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed("before <tool_call>\n")
	p.Feed(`{"name":"f","arguments":{"x":1}}`)
	p.Feed("\n</tool_call> after")
	p.Close()

	if spy.allProse() != "before  after" {
		t.Fatalf("got prose %q", spy.allProse())
	}
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "f" {
		t.Fatalf("tool starts: %+v", spy.toolStarts)
	}
	if spy.allArgs(0) != `{"x":1}` {
		t.Fatalf("got args %q", spy.allArgs(0))
	}
}

func TestStreamParser_CaseInsensitive(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`<TOOL_CALL>{"name":"f","arguments":{}}</TOOL_CALL>`)
	p.Close()
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "f" {
		t.Fatalf("case insensitive failed: %+v", spy.toolStarts)
	}
}

func TestStreamParser_InsideCodeFence_Ignored(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed("```\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\n```\n")
	p.Close()
	if len(spy.toolStarts) != 0 {
		t.Fatal("should not parse tool_call inside code fence")
	}
	if !strings.Contains(spy.allProse(), "<tool_call>") {
		t.Fatalf("fence content should be prose: %q", spy.allProse())
	}
}

func TestStreamParser_InsideInlineCode_Ignored(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed("Use `<tool_call>` to call tools")
	p.Close()
	if len(spy.toolStarts) != 0 {
		t.Fatal("should not parse tool_call inside inline code")
	}
	if spy.allProse() != "Use `<tool_call>` to call tools" {
		t.Fatalf("got prose %q", spy.allProse())
	}
}

func TestStreamParser_MultipleToolCalls(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`<tool_call>{"name":"a","arguments":{"x":1}}</tool_call><tool_call>{"name":"b","arguments":{"y":2}}</tool_call>`)
	p.Close()
	if len(spy.toolStarts) != 2 {
		t.Fatalf("expected 2 tool starts, got %d", len(spy.toolStarts))
	}
	if spy.toolStarts[0].name != "a" || spy.toolStarts[1].name != "b" {
		t.Fatalf("names: %+v", spy.toolStarts)
	}
}

func TestStreamParser_EmitsFirstToolBeforeSecondCompletes(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`<tool_call>{"name":"a","arguments":{}}</tool_call>`)
	if len(spy.toolEnds) != 1 || spy.toolEnds[0] != 0 {
		t.Fatalf("first tool should emit before second starts, ends=%+v starts=%+v", spy.toolEnds, spy.toolStarts)
	}
	p.Feed(`<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>`)
	p.Close()
	if len(spy.toolEnds) != 2 || spy.toolEnds[1] != 1 {
		t.Fatalf("second tool end missing: %+v", spy.toolEnds)
	}
}

func TestStreamParser_UnterminatedDegrade(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`Hello <tool_call>{"name":"f","arguments":{"x":`)
	p.Close()
	if !spy.completed {
		t.Fatal("should complete even on degrade")
	}
	combined := spy.allProse()
	if !strings.Contains(combined, "Hello") {
		t.Fatalf("should contain initial prose: %q", combined)
	}
}

func TestStreamParser_ClosingTagInsideJSON(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`<tool_call>{"name":"f","arguments":{"code":"</tool_call>"}}</tool_call>`)
	p.Close()
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "f" {
		t.Fatalf("tool starts: %+v", spy.toolStarts)
	}
	if spy.allArgs(0) != `{"code":"</tool_call>"}` {
		t.Fatalf("got args %q", spy.allArgs(0))
	}
}

func TestStreamParser_UTF8Boundary(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	// "中" is 3 bytes: 0xE4 0xB8 0xAD — split across two Feed calls
	b := []byte("中文")
	p.Feed(string(b[:1])) // incomplete rune
	p.Feed(string(b[1:])) // rest
	p.Close()
	if spy.allProse() != "中文" {
		t.Fatalf("got prose %q", spy.allProse())
	}
}

func TestStreamParser_DoesNotEmitToolCallBeforeArgumentsComplete(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})

	p.Feed(`<tool_call>{"name":"f","arguments":{"x":`)
	if len(spy.toolStarts) != 0 {
		t.Fatalf("tool start should wait for complete validated JSON, starts=%+v", spy.toolStarts)
	}
	if len(spy.argsDeltas) != 0 {
		t.Fatalf("args deltas should wait for complete validated JSON, deltas=%+v", spy.argsDeltas)
	}

	p.Feed(`1}}</tool_call>`)
	if len(spy.toolStarts) != 1 || spy.toolStarts[0].name != "f" {
		t.Fatalf("tool start after completion = %+v", spy.toolStarts)
	}
	if spy.allArgs(0) != `{"x":1}` {
		t.Fatalf("args = %q, want {\"x\":1}", spy.allArgs(0))
	}
	if len(spy.toolEnds) != 1 || spy.toolEnds[0] != 0 {
		t.Fatalf("tool end = %+v, want [0]", spy.toolEnds)
	}
}

func TestStreamParser_MalformedAfterArgumentsStartEmitsNoTool(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})

	p.Feed(`<tool_call>{"name":"f","arguments":{"x":1},"extra":`)
	p.Close()
	if len(spy.toolStarts) != 0 {
		t.Fatalf("malformed incomplete tool call must not emit a tool start: %+v", spy.toolStarts)
	}
	if len(spy.argsDeltas) != 0 {
		t.Fatalf("malformed incomplete tool call must not emit args: %+v", spy.argsDeltas)
	}
	if !strings.Contains(spy.allProse(), `<tool_call>`) {
		t.Fatalf("malformed call should degrade to prose, got %q", spy.allProse())
	}
}

func TestStreamParser_InvalidArgumentsDegradeToProse(t *testing.T) {
	spy := &spyEvents{}
	p := NewStreamParser(spy.events(), UpstreamMeta{ResponseID: "r", Provider: "p", Model: "m"})
	p.Feed(`<tool_call>{"name":"f","arguments":"not json"}</tool_call>`)
	p.Close()
	if len(spy.toolStarts) != 0 {
		t.Fatalf("invalid arguments should not emit tool call: %+v", spy.toolStarts)
	}
	if !strings.Contains(spy.allProse(), `<tool_call>`) {
		t.Fatalf("invalid call should degrade to prose, got %q", spy.allProse())
	}
}
