package toolemu

// mdContext tracks markdown code-fence and inline-code state to determine
// whether the current position is inside an escaped context where <tool_call>
// markers should be treated as literal text.
type mdContext struct {
	fenceBacktick int // odd = inside backtick fence
	fenceTilde    int // odd = inside tilde fence
	inlineTick    int // odd = inside inline code (resets on newline)
	atLineStart   bool
	pendingChar   rune // '`' or '~' being accumulated at line start
	pendingCount  int
}

func newMdContext() *mdContext {
	return &mdContext{atLineStart: true}
}

func (c *mdContext) inEscapedContext() bool {
	return c.fenceBacktick%2 == 1 || c.fenceTilde%2 == 1 || c.inlineTick%2 == 1
}

func (c *mdContext) feedRune(r rune) {
	if c.pendingCount > 0 && r == c.pendingChar {
		c.pendingCount++
		return
	}
	if c.pendingCount > 0 {
		c.flushPending()
	}
	c.processRune(r)
}

func (c *mdContext) flushPending() {
	inFence := c.fenceBacktick%2 == 1 || c.fenceTilde%2 == 1
	if c.pendingCount >= 3 {
		switch c.pendingChar {
		case '`':
			// open or close backtick fence (only if not inside a tilde fence)
			if !inFence || c.fenceBacktick%2 == 1 {
				c.fenceBacktick++
				c.inlineTick = 0
			}
		case '~':
			// open or close tilde fence (only if not inside a backtick fence)
			if !inFence || c.fenceTilde%2 == 1 {
				c.fenceTilde++
			}
		}
	} else if c.pendingChar == '`' && !inFence {
		// 1 or 2 backticks at line start → inline code toggles
		c.inlineTick += c.pendingCount
	}
	c.pendingCount = 0
	c.pendingChar = 0
	c.atLineStart = false
}

func (c *mdContext) processRune(r rune) {
	inFence := c.fenceBacktick%2 == 1 || c.fenceTilde%2 == 1

	switch {
	case r == '\n':
		c.atLineStart = true
		if !inFence {
			c.inlineTick = 0
		}
	case r == '`' && c.atLineStart:
		c.pendingChar = '`'
		c.pendingCount = 1
	case r == '~' && c.atLineStart:
		c.pendingChar = '~'
		c.pendingCount = 1
	case r == '`' && !inFence:
		// inline code toggle (not at line start, not in fence)
		c.inlineTick++
		c.atLineStart = false
	default:
		if r != ' ' && r != '\t' {
			c.atLineStart = false
		}
	}
}

// flush finalizes any pending fence sequence at end of input.
func (c *mdContext) flush() {
	if c.pendingCount > 0 {
		c.flushPending()
	}
}
