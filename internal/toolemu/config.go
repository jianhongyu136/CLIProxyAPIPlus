package toolemu

// ToolEmulationConfig binds the top-level `tool-emulation` block in config.yaml.
type ToolEmulationConfig struct {
	Enabled        bool                `yaml:"enabled" json:"enabled"`
	Rules          []ToolEmulationRule `yaml:"rules" json:"rules"`
	ParseRetry     int                 `yaml:"parse-retry" json:"parse-retry"`
	OnParseFailure string              `yaml:"on-parse-failure" json:"on-parse-failure"`
}

// ToolEmulationRule is a single first-match-wins gating rule.
type ToolEmulationRule struct {
	Provider     string   `yaml:"provider" json:"provider"`
	Models       []string `yaml:"models" json:"models"`
	ModelAliases []string `yaml:"model-aliases" json:"model-aliases"`
}

// DefaultsApplied returns a copy with defaults filled in for unset fields.
func (c ToolEmulationConfig) DefaultsApplied() ToolEmulationConfig {
	out := c
	if out.OnParseFailure == "" {
		out.OnParseFailure = "parse_failed_to_content"
	}
	return out
}

// DefaultParseRetry is applied when the parse-retry field is absent.
const DefaultParseRetry = 1
