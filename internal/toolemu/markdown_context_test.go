package toolemu

import "testing"

func TestMdContext_BacktickFence(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "```\n<tool_call>\n```" {
		ctx.feedRune(r)
	}
	ctx.flush()
	if ctx.inEscapedContext() {
		t.Fatal("should not be escaped after closing fence")
	}
}

func TestMdContext_BacktickFence_InsideIsEscaped(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "```\n" {
		ctx.feedRune(r)
	}
	if !ctx.inEscapedContext() {
		t.Fatal("should be escaped inside backtick fence")
	}
}

func TestMdContext_TildeFence(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "~~~\n" {
		ctx.feedRune(r)
	}
	if !ctx.inEscapedContext() {
		t.Fatal("should be escaped inside tilde fence")
	}
	for _, r := range "some code\n~~~\n" {
		ctx.feedRune(r)
	}
	ctx.flush()
	if ctx.inEscapedContext() {
		t.Fatal("should not be escaped after closing tilde fence")
	}
}

func TestMdContext_InlineCode(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "text `" {
		ctx.feedRune(r)
	}
	if !ctx.inEscapedContext() {
		t.Fatal("should be escaped inside inline code")
	}
	for _, r := range "<tool_call>`" {
		ctx.feedRune(r)
	}
	if ctx.inEscapedContext() {
		t.Fatal("should not be escaped after closing backtick")
	}
}

func TestMdContext_InlineCodeResetsOnNewline(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "text `unclosed\n" {
		ctx.feedRune(r)
	}
	if ctx.inEscapedContext() {
		t.Fatal("inline code should reset on newline")
	}
}

func TestMdContext_FenceRequiresLineStart(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "text ```\n" {
		ctx.feedRune(r)
	}
	ctx.flush()
	if ctx.inEscapedContext() {
		t.Fatal("``` not at line start should not open fence")
	}
}

func TestMdContext_FenceWithInfoString(t *testing.T) {
	ctx := newMdContext()
	for _, r := range "```go\ncode\n```\n" {
		ctx.feedRune(r)
	}
	ctx.flush()
	if ctx.inEscapedContext() {
		t.Fatal("should not be escaped after fence with info string closes")
	}
}
