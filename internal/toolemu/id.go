package toolemu

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// DeriveID returns a deterministic, collision-resistant tool_call id from
// upstream metadata and the call's index within the response.
//
// When meta.ResponseID is empty, falls back to random 8 hex chars (still
// prefixed with "call_") — this should be rare in practice.
func DeriveID(meta UpstreamMeta, index int) string {
	return deriveIDWithPrefix(meta, index, "call_")
}

// DeriveClaudeID returns a deterministic Claude tool_use id.
func DeriveClaudeID(meta UpstreamMeta, index int) string {
	return deriveIDWithPrefix(meta, index, "toolu_")
}

func deriveIDWithPrefix(meta UpstreamMeta, index int, prefix string) string {
	if meta.ResponseID == "" {
		var b [4]byte
		_, _ = rand.Read(b[:])
		return prefix + hex.EncodeToString(b[:])
	}
	h := sha256.New()
	fmt.Fprintf(h, "%s\x1f%s\x1f%s\x1f%d", meta.Provider, meta.Model, meta.ResponseID, index)
	return prefix + hex.EncodeToString(h.Sum(nil)[:4])
}
