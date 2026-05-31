package toolemu

import (
	"path/filepath"
	"sync/atomic"

	log "github.com/sirupsen/logrus"
)

// Matcher resolves whether toolemu should fire for a given (provider, model, alias) tuple.
//
// It is safe for concurrent use. Replace the underlying snapshot atomically when
// configuration is hot-reloaded.
type Matcher struct {
	snap atomic.Pointer[matcherSnap]
}

type matcherSnap struct {
	enabled bool
	rules   []compiledRule
}

type compiledRule struct {
	provider string
	models   []string // raw glob patterns; matched via filepath.Match
	aliases  []string
}

// NewMatcher compiles cfg into an immutable snapshot.
func NewMatcher(cfg ToolEmulationConfig) *Matcher {
	m := &Matcher{}
	m.Replace(cfg)
	return m
}

// Replace atomically swaps the active snapshot.
func (m *Matcher) Replace(cfg ToolEmulationConfig) {
	snap := &matcherSnap{enabled: cfg.Enabled}
	for _, r := range cfg.Rules {
		snap.rules = append(snap.rules, compiledRule{
			provider: r.Provider,
			models:   append([]string(nil), r.Models...),
			aliases:  append([]string(nil), r.ModelAliases...),
		})
	}
	prev := m.snap.Swap(snap)

	// Log when the toggle or rule count changes so operators can confirm a
	// hot-reload actually flipped toolemu on or off.
	prevEnabled := false
	prevRules := 0
	if prev != nil {
		prevEnabled = prev.enabled
		prevRules = len(prev.rules)
	}
	if cfg.Enabled != prevEnabled || len(snap.rules) != prevRules {
		if cfg.Enabled {
			log.Infof("toolemu: enabled (rules=%d)", len(snap.rules))
		} else if prevEnabled {
			log.Infof("toolemu: disabled")
		}
	}
}

// IsEnabled returns true when the first matching rule applies.
func (m *Matcher) IsEnabled(provider, model, alias string) bool {
	snap := m.snap.Load()
	if snap == nil || !snap.enabled {
		return false
	}
	for _, r := range snap.rules {
		if r.provider != provider {
			continue
		}
		if len(r.models) == 0 && len(r.aliases) == 0 {
			return true
		}
		if matchesAny(r.models, model) {
			return true
		}
		if alias != "" && matchesAny(r.aliases, alias) {
			return true
		}
	}
	return false
}

func matchesAny(patterns []string, s string) bool {
	for _, p := range patterns {
		ok, err := filepath.Match(p, s)
		if err == nil && ok {
			return true
		}
	}
	return false
}

// Default is the package-level matcher used by runtime executors.
//
// Call Default.Replace(cfg.ToolEmulation) on startup and on config reload.
var Default = NewMatcher(ToolEmulationConfig{})
