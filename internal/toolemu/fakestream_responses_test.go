package toolemu

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestFakeStreamResponses_OrderAndStructure(t *testing.T) {
	p := Parsed{ToolCalls: []ParsedToolCall{{
		Name: "f", Arguments: []byte(strings.Repeat(`"`, argChunkSize+5)),
	}}}
	chunks, err := FakeStreamResponses(p, UpstreamMeta{ResponseID: "r", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("no chunks emitted")
	}

	// Each chunk must have "event: " and "data: " lines.
	for i, c := range chunks {
		if !bytes.HasPrefix(c, []byte("event: ")) {
			t.Fatalf("chunk %d missing event: prefix: %q", i, c)
		}
		if !bytes.Contains(c, []byte("\ndata: ")) {
			t.Fatalf("chunk %d missing data: line: %q", i, c)
		}
		if !bytes.HasSuffix(c, []byte("\n\n")) {
			t.Fatalf("chunk %d missing trailing \\n\\n: %q", i, c)
		}
	}

	want := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	seenAt := map[string]int{}
	for i, c := range chunks {
		for _, name := range want {
			if bytes.HasPrefix(c, []byte("event: "+name+"\n")) {
				if _, ok := seenAt[name]; !ok {
					seenAt[name] = i
				}
			}
		}
	}
	prev := -1
	for _, name := range want {
		idx, ok := seenAt[name]
		if !ok {
			t.Fatalf("missing event %q", name)
		}
		if idx <= prev {
			t.Fatalf("event %q at %d came before previous expected event (prev=%d)", name, idx, prev)
		}
		prev = idx
	}
}

func TestFakeStreamResponses_Deterministic(t *testing.T) {
	p := Parsed{Prose: "x", ToolCalls: []ParsedToolCall{{Name: "f", Arguments: []byte(`{"a":1}`)}}}
	meta := UpstreamMeta{ResponseID: "r", Model: "m"}
	a, _ := FakeStreamResponses(p, meta)
	b, _ := FakeStreamResponses(p, meta)
	if len(a) != len(b) {
		t.Fatalf("len mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			t.Fatalf("frame %d differs\nA: %s\nB: %s", i, a[i], b[i])
		}
	}
}

func TestFakeStreamResponses_PreservesUTF8AcrossChunkBoundaries(t *testing.T) {
	prose := strings.Repeat("a", proseChunkSize-1) + "中文结尾"
	argPrefix := strings.Repeat("a", argChunkSize-1-len(`{"text":"`))
	args := []byte(`{"text":"` + argPrefix + `中文参数"}`)
	p := Parsed{
		Prose: prose,
		ToolCalls: []ParsedToolCall{{
			Name: "f", Arguments: args,
		}},
	}

	chunks, err := FakeStreamResponses(p, UpstreamMeta{ResponseID: "r", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}

	var gotProse strings.Builder
	var gotArgs strings.Builder
	for _, c := range chunks {
		eventType, body := responseEvent(c)
		switch eventType {
		case "response.output_text.delta":
			gotProse.WriteString(gjson.GetBytes(body, "delta").String())
		case "response.function_call_arguments.delta":
			gotArgs.WriteString(gjson.GetBytes(body, "delta").String())
		}
	}

	if gotProse.String() != prose {
		t.Fatalf("prose mismatch\n got: %q\nwant: %q", gotProse.String(), prose)
	}
	if gotArgs.String() != string(args) {
		t.Fatalf("arguments mismatch\n got: %q\nwant: %q", gotArgs.String(), string(args))
	}
}

func responseEvent(chunk []byte) (string, []byte) {
	parts := bytes.SplitN(chunk, []byte("\ndata: "), 2)
	if len(parts) != 2 {
		return "", nil
	}
	eventType := string(bytes.TrimPrefix(parts[0], []byte("event: ")))
	body := bytes.TrimSuffix(parts[1], []byte("\n\n"))
	return eventType, body
}

func jsonStringField(body []byte, field string) string {
	return gjson.GetBytes(body, field).String()
}
