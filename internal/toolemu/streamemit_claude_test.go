package toolemu

import (
	"bytes"
	"testing"
)

func TestClaudeStreamEmitterIncludesEventLines(t *testing.T) {
	var frames [][]byte
	emitter := NewClaudeStreamEmitter(UpstreamMeta{Provider: "p", Model: "m", ResponseID: "msg_1"}, func(frame []byte) {
		frames = append(frames, bytes.Clone(frame))
	})
	events := emitter.Events()
	events.OnProseDelta("hello")
	events.OnComplete()

	joined := bytes.Join(frames, nil)
	if !bytes.Contains(joined, []byte("event: message_start\n")) {
		t.Fatalf("missing message_start event line:\n%s", joined)
	}
	if !bytes.Contains(joined, []byte("event: content_block_delta\n")) {
		t.Fatalf("missing content_block_delta event line:\n%s", joined)
	}
	if !bytes.Contains(joined, []byte("event: message_stop\n")) {
		t.Fatalf("missing message_stop event line:\n%s", joined)
	}
}
