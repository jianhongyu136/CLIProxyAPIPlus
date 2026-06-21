package toolemu

import (
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/tidwall/gjson"
)

// StreamEvents defines the callbacks emitted by StreamParser.
type StreamEvents struct {
	OnMetaUpdate     func(meta UpstreamMeta)
	OnProseDelta     func(delta string)
	OnReasoningDelta func(delta string)
	OnToolCallStart  func(index int, id, name string)
	OnArgsDelta      func(index int, delta string)
	OnToolCallEnd    func(index int)
	OnComplete       func()
	OnError          func(err error)
}

type streamState int

const (
	stProse     streamState = iota
	stMaybeOpen             // buffering potential "<tool_call>" prefix
	stInCall                // inside tool_call block, parsing JSON
	stSkipClose             // consuming optional </tool_call> after JSON closes
)

// StreamParser is an incremental state machine that parses <tool_call> blocks
// from a stream of text chunks. Feed arbitrary-sized string fragments via
// Feed() and receive structured events through StreamEvents callbacks.
type StreamParser struct {
	events StreamEvents
	meta   UpstreamMeta
	md     *mdContext

	state     streamState
	buf       strings.Builder // buffer for MaybeOpen or SkipClose
	toolIndex int

	// InCall sub-state: JSON depth tracking
	jsonDepth int
	inStr     bool
	escape    bool
	headBuf   strings.Builder // accumulates full JSON object

	toolStarted bool
	argStarted  bool
	argDepth    int
	argInStr    bool
	argEscape   bool
	argBuf      strings.Builder

	// UTF-8 partial rune buffer
	pendingBytes []byte
}

// NewStreamParser creates a parser that emits events as text is fed.
func NewStreamParser(events StreamEvents, meta UpstreamMeta) *StreamParser {
	return &StreamParser{
		events: events,
		meta:   meta,
		md:     newMdContext(),
	}
}

// Feed processes a chunk of text from the upstream stream.
func (p *StreamParser) Feed(chunk string) {
	// Prepend any pending incomplete UTF-8 bytes
	var data []byte
	if len(p.pendingBytes) > 0 {
		data = append(p.pendingBytes, []byte(chunk)...)
		p.pendingBytes = p.pendingBytes[:0]
	} else {
		data = []byte(chunk)
	}
	valid := validUTF8Prefix(data)
	if valid < len(data) {
		p.pendingBytes = append(p.pendingBytes, data[valid:]...)
		data = data[:valid]
	}
	for _, r := range string(data) {
		p.feedRune(r)
	}
}

// Close signals end of stream. Flushes buffered content and calls OnComplete.
func (p *StreamParser) Close() {
	// Flush any pending incomplete UTF-8 bytes as-is
	if len(p.pendingBytes) > 0 {
		for _, r := range string(p.pendingBytes) {
			p.feedRune(r)
		}
		p.pendingBytes = nil
	}
	p.md.flush()

	switch p.state {
	case stMaybeOpen:
		p.emitProse(p.buf.String())
		p.buf.Reset()
	case stInCall:
		p.degradeInCall()
	case stSkipClose:
		flush := p.buf.String()
		p.buf.Reset()
		if strings.TrimSpace(flush) != "" {
			for i, fr := range flush {
				if i == 0 {
					p.emitProseRune(fr)
					continue
				}
				p.md.feedRune(fr)
				p.emitProseRune(fr)
			}
		}
	}

	if p.events.OnComplete != nil {
		p.events.OnComplete()
	}
}

// UpdateMeta refreshes metadata learned from upstream frames before callbacks emit downstream frames.
func (p *StreamParser) UpdateMeta(meta UpstreamMeta) {
	p.meta = meta
	if p.events.OnMetaUpdate != nil {
		p.events.OnMetaUpdate(meta)
	}
}

func (p *StreamParser) feedRune(r rune) {
	switch p.state {
	case stProse:
		p.feedProse(r)
	case stMaybeOpen:
		p.feedMaybeOpen(r)
	case stInCall:
		p.feedInCall(r)
	case stSkipClose:
		p.feedSkipClose(r)
	}
}

func (p *StreamParser) feedProse(r rune) {
	p.md.feedRune(r)
	if r == '<' && !p.md.inEscapedContext() {
		p.state = stMaybeOpen
		p.buf.Reset()
		p.buf.WriteRune(r)
		return
	}
	p.emitProseRune(r)
}

func (p *StreamParser) feedMaybeOpen(r rune) {
	p.buf.WriteRune(r)
	bufStr := p.buf.String()

	if len(bufStr) <= len(openTag) && strings.EqualFold(bufStr, openTag[:len(bufStr)]) {
		if strings.EqualFold(bufStr, openTag) {
			p.state = stInCall
			p.buf.Reset()
			p.resetInCallState()
		}
		return
	}

	// Not a match — flush buffer as prose
	p.state = stProse
	flush := p.buf.String()
	p.buf.Reset()
	// First char '<' was already fed to md context in feedProse; just emit it.
	// Remaining chars need md context updates.
	for i, fr := range flush {
		if i == 0 {
			p.emitProseRune(fr)
			continue
		}
		p.md.feedRune(fr)
		if fr == '<' && !p.md.inEscapedContext() {
			// This '<' might start another tag — re-enter MaybeOpen
			p.state = stMaybeOpen
			p.buf.Reset()
			p.buf.WriteRune(fr)
			// Emit remaining chars through feedRune
			rest := flush[i+utf8.RuneLen(fr):]
			for _, rr := range rest {
				p.feedRune(rr)
			}
			return
		}
		p.emitProseRune(fr)
	}
}

func (p *StreamParser) resetInCallState() {
	p.jsonDepth = 0
	p.inStr = false
	p.escape = false
	p.headBuf.Reset()
	p.toolStarted = false
	p.argStarted = false
	p.argDepth = 0
	p.argInStr = false
	p.argEscape = false
	p.argBuf.Reset()
}

func (p *StreamParser) feedInCall(r rune) {
	// Skip leading whitespace before JSON object.
	if p.jsonDepth == 0 && !p.inStr && isASCIISpace(byte(r)) {
		return
	}

	p.headBuf.WriteRune(r)

	if p.inStr {
		if p.escape {
			p.escape = false
		} else if r == '\\' {
			p.escape = true
		} else if r == '"' {
			p.inStr = false
		}
		return
	}

	switch r {
	case '"':
		p.inStr = true
	case '{', '[':
		p.jsonDepth++
	case '}', ']':
		p.jsonDepth--
		if p.jsonDepth == 0 {
			p.finishToolCall()
		}
	}
}

func (p *StreamParser) maybeStartToolCall() {
	if p.toolStarted {
		return
	}
	raw := p.headBuf.String()
	name := normalizeToolName(gjson.Get(raw, "name").String())
	if name == "" || !argumentsObjectStarted(raw) {
		return
	}
	p.toolStarted = true
	id := DeriveID(p.meta, p.toolIndex)
	if p.events.OnToolCallStart != nil {
		p.events.OnToolCallStart(p.toolIndex, id, name)
	}
}

func argumentsObjectStarted(raw string) bool {
	idx := strings.Index(raw, `"arguments"`)
	if idx < 0 {
		return false
	}
	idx += len(`"arguments"`)
	for idx < len(raw) && isASCIISpace(raw[idx]) {
		idx++
	}
	if idx >= len(raw) || raw[idx] != ':' {
		return false
	}
	idx++
	for idx < len(raw) && isASCIISpace(raw[idx]) {
		idx++
	}
	return idx < len(raw) && raw[idx] == '{'
}

func (p *StreamParser) feedArgumentRune(r rune) {
	if !p.toolStarted {
		return
	}
	if !p.argStarted {
		raw := p.headBuf.String()
		idx := strings.Index(raw, `"arguments"`)
		if idx < 0 {
			return
		}
		idx += len(`"arguments"`)
		for idx < len(raw) && isASCIISpace(raw[idx]) {
			idx++
		}
		if idx >= len(raw) || raw[idx] != ':' {
			return
		}
		idx++
		for idx < len(raw) && isASCIISpace(raw[idx]) {
			idx++
		}
		if idx >= len(raw) || raw[idx] != '{' {
			return
		}
		p.argStarted = true
		p.argDepth = 0
		p.argInStr = false
		p.argEscape = false
		p.argBuf.Reset()
		for _, rr := range raw[idx:] {
			p.appendArgumentRune(rr)
		}
		return
	}
	p.appendArgumentRune(r)
}

func (p *StreamParser) appendArgumentRune(r rune) {
	if p.argDepth == 0 && p.argBuf.Len() > 0 {
		return
	}
	p.argBuf.WriteRune(r)
	if p.argInStr {
		if p.argEscape {
			p.argEscape = false
		} else if r == '\\' {
			p.argEscape = true
		} else if r == '"' {
			p.argInStr = false
		}
	} else {
		switch r {
		case '"':
			p.argInStr = true
		case '{':
			p.argDepth++
		case '}':
			p.argDepth--
		}
	}
	if p.events.OnArgsDelta != nil {
		p.events.OnArgsDelta(p.toolIndex, string(r))
	}
}

func (p *StreamParser) finishToolCall() {
	raw := p.headBuf.String()
	p.headBuf.Reset()

	name, args, ok := extractNameAndArgs(raw)
	if !ok {
		// Degrade: emit as prose
		p.emitProse(openTag + raw)
		p.state = stProse
		return
	}

	if !p.toolStarted {
		id := DeriveID(p.meta, p.toolIndex)
		if p.events.OnToolCallStart != nil {
			p.events.OnToolCallStart(p.toolIndex, id, name)
		}
		if p.events.OnArgsDelta != nil && args != "" {
			p.events.OnArgsDelta(p.toolIndex, args)
		}
	}
	if p.events.OnToolCallEnd != nil {
		p.events.OnToolCallEnd(p.toolIndex)
	}
	p.toolIndex++

	// Switch to skip-close state to consume optional </tool_call>
	p.buf.Reset()
	p.state = stSkipClose
}

func (p *StreamParser) degradeInCall() {
	raw := p.headBuf.String()
	p.headBuf.Reset()
	p.emitProse(openTag + raw)
	p.state = stProse
}

func (p *StreamParser) feedSkipClose(r rune) {
	p.buf.WriteRune(r)
	bufStr := p.buf.String()
	trimmed := strings.TrimLeft(bufStr, " \t\n\r")

	if trimmed == "" {
		return
	}
	if len(trimmed) <= len(closeTag) && strings.EqualFold(trimmed, closeTag[:len(trimmed)]) {
		if strings.EqualFold(trimmed, closeTag) {
			p.buf.Reset()
			p.state = stProse
		}
		return
	}

	p.state = stProse
	flush := p.buf.String()
	p.buf.Reset()
	if strings.TrimSpace(flush) == "" {
		return
	}
	for _, fr := range flush {
		p.feedRune(fr)
	}
}

func (p *StreamParser) emitProse(s string) {
	if s == "" || p.events.OnProseDelta == nil {
		return
	}
	p.events.OnProseDelta(s)
}

func (p *StreamParser) emitProseRune(r rune) {
	if p.events.OnProseDelta != nil {
		p.events.OnProseDelta(string(r))
	}
}

// extractNameAndArgs parses a complete JSON object and returns the "name" field
// value and the normalized "arguments" value as a string.
func extractNameAndArgs(raw string) (name, args string, ok bool) {
	raw = strings.TrimSpace(raw)
	obj, err := tryParseJSONObject([]byte(raw))
	if err != nil {
		return "", "", false
	}
	nameRaw, exists := obj["name"]
	if !exists {
		return "", "", false
	}
	var n string
	if err := json.Unmarshal(nameRaw, &n); err != nil || n == "" {
		return "", "", false
	}
	name = normalizeToolName(n)
	if name == "" {
		return "", "", false
	}

	argsBytes, errArgs := normalizeArguments(obj["arguments"])
	if errArgs != nil {
		return "", "", false
	}
	args = string(argsBytes)
	return name, args, true
}

// validUTF8Prefix returns the length of the longest valid UTF-8 prefix of b.
func validUTF8Prefix(b []byte) int {
	n := len(b)
	if n == 0 || utf8.Valid(b) {
		return n
	}
	// An incomplete rune is at most 3 trailing bytes; check 1-3 from the end.
	for i := 1; i <= 3 && i < n; i++ {
		if utf8.Valid(b[:n-i]) {
			return n - i
		}
	}
	return 0
}
