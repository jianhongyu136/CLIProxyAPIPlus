package kiro

import (
	"sync"
	"time"

	log "github.com/sirupsen/logrus"
)

var (
	globalRateLimiter     *RateLimiter
	globalRateLimiterOnce sync.Once
	globalRateLimiterCfg  *RateLimiterConfig

	globalCooldownManager     *CooldownManager
	globalCooldownManagerOnce sync.Once
	cooldownStopCh            chan struct{}

	// Global retry config values read by executor
	globalMaxRetriesOn429    int
	globalRetryDelayOn429    time.Duration
	globalMaxEndpointRetries int
)

// SetGlobalRateLimiterConfig sets the configuration for the global rate limiter.
// If the singleton already exists, it is updated in place.
func SetGlobalRateLimiterConfig(cfg *RateLimiterConfig) {
	globalRateLimiterCfg = cfg
	if globalRateLimiter == nil {
		applyRetryConfig(cfg)
		return
	}

	if cfg != nil {
		globalRateLimiter.ApplyConfig(*cfg)
	} else {
		globalRateLimiter.ApplyConfig(RateLimiterConfig{})
	}

	applyRetryConfig(cfg)

	status := "enabled"
	if !globalRateLimiter.enabled {
		status = "disabled"
	}
	source := "defaults"
	if cfg != nil {
		source = "custom config"
	}
	log.Infof("kiro: global RateLimiter reconfigured (%s) with %s", status, source)
}

// applyRetryConfig applies retry/cooldown config from RateLimiterConfig to global state.
func applyRetryConfig(cfg *RateLimiterConfig) {
	// Reset to defaults
	globalMaxRetriesOn429 = 0
	globalRetryDelayOn429 = 2 * time.Second
	globalMaxEndpointRetries = 2

	if cfg == nil {
		return
	}

	// Apply cooldown enabled/disabled to the CooldownManager
	if cfg.CooldownEnabled != nil {
		cooldownDisabled := !*cfg.CooldownEnabled
		cm := GetGlobalCooldownManager()
		cm.SetDisabled(cooldownDisabled)
		if cooldownDisabled {
			log.Infof("kiro: CooldownManager disabled via config")
		}
	}

	if cfg.MaxRetriesOn429 > 0 {
		globalMaxRetriesOn429 = cfg.MaxRetriesOn429
		log.Infof("kiro: max-retries-on-429 set to %d", globalMaxRetriesOn429)
	}
	if cfg.RetryDelayOn429 > 0 {
		globalRetryDelayOn429 = cfg.RetryDelayOn429
		log.Infof("kiro: retry-delay-on-429 set to %v", globalRetryDelayOn429)
	}
	if cfg.MaxEndpointRetries > 0 {
		globalMaxEndpointRetries = cfg.MaxEndpointRetries
		log.Infof("kiro: max-endpoint-retries set to %d", globalMaxEndpointRetries)
	}
}

// GetMaxRetriesOn429 returns the configured max retries for 429 errors per endpoint.
func GetMaxRetriesOn429() int {
	return globalMaxRetriesOn429
}

// GetRetryDelayOn429 returns the configured delay between 429 retries.
func GetRetryDelayOn429() time.Duration {
	return globalRetryDelayOn429
}

// GetMaxEndpointRetries returns the configured max retries per endpoint.
func GetMaxEndpointRetries() int {
	return globalMaxEndpointRetries
}

// GetGlobalRateLimiter returns the singleton RateLimiter instance.
func GetGlobalRateLimiter() *RateLimiter {
	globalRateLimiterOnce.Do(func() {
		if globalRateLimiterCfg != nil {
			globalRateLimiter = NewRateLimiterWithConfig(*globalRateLimiterCfg)
		} else {
			globalRateLimiter = NewRateLimiter()
		}
		status := "enabled"
		if !globalRateLimiter.enabled {
			status = "disabled"
		}
		source := "defaults"
		if globalRateLimiterCfg != nil {
			source = "custom config"
		}
		log.Infof("kiro: global RateLimiter initialized (%s) with %s", status, source)
	})
	return globalRateLimiter
}

// GetGlobalCooldownManager returns the singleton CooldownManager instance.
func GetGlobalCooldownManager() *CooldownManager {
	globalCooldownManagerOnce.Do(func() {
		globalCooldownManager = NewCooldownManager()
		cooldownStopCh = make(chan struct{})
		go globalCooldownManager.StartCleanupRoutine(5*time.Minute, cooldownStopCh)
		log.Info("kiro: global CooldownManager initialized with cleanup routine")
	})
	return globalCooldownManager
}

// ShutdownRateLimiters stops the cooldown cleanup routine.
// Should be called during application shutdown.
func ShutdownRateLimiters() {
	if cooldownStopCh != nil {
		close(cooldownStopCh)
		log.Info("kiro: rate limiter cleanup routine stopped")
	}
}
