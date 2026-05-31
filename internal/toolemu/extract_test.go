package toolemu

import (
	"bytes"
	"testing"
)

func TestExtractAssistantText_Chat(t *testing.T) {
	body := []byte(`{
		"id":"r1","model":"m","created":1,
		"choices":[{"message":{"role":"assistant","content":"hello"}}],
		"usage":{"x":1}
	}`)
	text, meta, err := ExtractAssistantText(body, ShapeOpenAIChat)
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("text = %q", text)
	}
	if meta.ResponseID != "r1" || meta.Model != "m" || meta.Created != 1 {
		t.Fatalf("meta = %+v", meta)
	}
	if !bytes.Contains(meta.UsagePayload, []byte(`"x":1`)) {
		t.Fatalf("usage payload missing: %s", meta.UsagePayload)
	}
}

func TestExtractAssistantText_Responses(t *testing.T) {
	body := []byte(`{
		"id":"r2","model":"m2","created_at":2,
		"output":[{"type":"message","role":"assistant","content":[
			{"type":"output_text","text":"a"},
			{"type":"output_text","text":"b"}
		]}],
		"usage":{"y":3}
	}`)
	text, meta, err := ExtractAssistantText(body, ShapeOpenAIResponses)
	if err != nil {
		t.Fatal(err)
	}
	if text != "ab" {
		t.Fatalf("text = %q want ab", text)
	}
	if meta.ResponseID != "r2" || meta.Model != "m2" || meta.Created != 2 {
		t.Fatalf("meta = %+v", meta)
	}
	if !bytes.Contains(meta.UsagePayload, []byte(`"y":3`)) {
		t.Fatalf("usage payload missing: %s", meta.UsagePayload)
	}
}

func TestExtractAssistantText_UnknownShapeErrors(t *testing.T) {
	if _, _, err := ExtractAssistantText([]byte(`{}`), UpstreamShape(99)); err == nil {
		t.Fatal("expected error for unknown shape")
	}
}
