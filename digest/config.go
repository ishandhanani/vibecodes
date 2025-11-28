package main

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// WatchType defines what kind of activity to track
type WatchType string

const (
	WatchMentions WatchType = "mentions" // issues/PRs involving the user
	WatchNewPRs   WatchType = "new_prs"  // recently opened PRs
	WatchIssues   WatchType = "issues"   // issues (optionally with keywords)
	WatchLabeled  WatchType = "labeled"  // items with specific label
	WatchTopics   WatchType = "topics"   // items matching repo's interests
)

// Watch defines a single watch configuration
type Watch struct {
	Type     WatchType `yaml:"type"`
	Keywords []string  `yaml:"keywords,omitempty"` // for issues watch
	Label    string    `yaml:"label,omitempty"`    // for labeled watch
}

// RepoConfig defines a repository to watch
type RepoConfig struct {
	Name    string   `yaml:"name"`
	Repo    string   `yaml:"repo"`              // owner/repo format
	Topics  []string `yaml:"topics,omitempty"`  // topics to track (each becomes a section)
	Watches []Watch  `yaml:"watches,omitempty"` // additional watches
	Exclude []string `yaml:"exclude,omitempty"` // topics to deprioritize/filter out
}

// AIConfig defines AI analysis settings
type AIConfig struct {
	Enabled bool   `yaml:"enabled"`
	Model   string `yaml:"model"`
}

// Config is the root configuration
type Config struct {
	User     string       `yaml:"user"`
	DaysBack int          `yaml:"days_back"`
	Exclude  []string     `yaml:"exclude,omitempty"` // global exclusions (topics to ignore)
	Repos    []RepoConfig `yaml:"repos"`
	AI       AIConfig     `yaml:"ai"`
}

// DefaultConfig returns a sensible default configuration
func DefaultConfig() Config {
	return Config{
		User:     "ishandhanani",
		DaysBack: 1,
		Repos: []RepoConfig{
			{
				Name: "SGLang",
				Repo: "sgl-project/sglang",
				Topics: []string{
					"nixl",
					"p/d disaggregation",
					"tracing",
					"metrics",
					"dp attention",
				},
			},
			{
				Name:   "Dynamo",
				Repo:   "ai-dynamo/dynamo",
				Topics: []string{"sglang"},
				Watches: []Watch{
					{Type: WatchLabeled, Label: "external contribution"},
				},
			},
		},
		AI: AIConfig{
			Enabled: true,
			Model:   "openai/gpt-4o-mini",
		},
	}
}

// configDir returns the config directory path
func configDir() string {
	if d := os.Getenv("DIGEST_CONFIG_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "digest")
}

// configPath returns the full path to config.yaml
func configPath() string {
	return filepath.Join(configDir(), "config.yaml")
}

// LoadConfig loads config from default location, or returns default if not found
func LoadConfig() (Config, error) {
	return LoadConfigFrom(configPath())
}

// LoadConfigFrom loads config from a specific file path
func LoadConfigFrom(path string) (Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Create default config file at default location
			cfg := DefaultConfig()
			if path == configPath() {
				if err := SaveConfig(cfg); err != nil {
					return cfg, nil // still return default, just couldn't save
				}
			}
			return cfg, nil
		}
		return Config{}, fmt.Errorf("reading config: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("parsing config: %w", err)
	}

	// Apply defaults for missing fields
	if cfg.DaysBack == 0 {
		cfg.DaysBack = 1
	}
	if cfg.AI.Model == "" {
		cfg.AI.Model = "openai/gpt-4o-mini"
	}

	return cfg, nil
}

// AllInterests returns all topics from all repos (deduplicated)
func (c Config) AllInterests() []string {
	seen := make(map[string]bool)
	var all []string
	for _, repo := range c.Repos {
		for _, t := range repo.Topics {
			if !seen[t] {
				seen[t] = true
				all = append(all, t)
			}
		}
	}
	return all
}

// AllExclusions returns all exclusions from global + all repos (deduplicated)
func (c Config) AllExclusions() []string {
	seen := make(map[string]bool)
	var all []string
	for _, e := range c.Exclude {
		if !seen[e] {
			seen[e] = true
			all = append(all, e)
		}
	}
	for _, repo := range c.Repos {
		for _, e := range repo.Exclude {
			if !seen[e] {
				seen[e] = true
				all = append(all, e)
			}
		}
	}
	return all
}

// SaveConfig saves config to the config file
func SaveConfig(cfg Config) error {
	if err := os.MkdirAll(configDir(), 0o755); err != nil {
		return err
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(configPath(), data, 0o644)
}

// Validate checks the config for errors
func (c Config) Validate() error {
	if c.User == "" {
		return fmt.Errorf("user is required")
	}
	if len(c.Repos) == 0 {
		return fmt.Errorf("at least one repo is required")
	}
	for _, repo := range c.Repos {
		if repo.Repo == "" {
			return fmt.Errorf("repo %q has no repo path", repo.Name)
		}
		if len(repo.Topics) == 0 && len(repo.Watches) == 0 {
			return fmt.Errorf("repo %q has no topics or watches", repo.Name)
		}
		for _, w := range repo.Watches {
			switch w.Type {
			case WatchMentions, WatchNewPRs, WatchTopics:
				// no extra fields required
			case WatchIssues:
				// keywords optional
			case WatchLabeled:
				if w.Label == "" {
					return fmt.Errorf("repo %q: labeled watch requires label", repo.Name)
				}
			default:
				return fmt.Errorf("repo %q: unknown watch type %q", repo.Name, w.Type)
			}
		}
	}
	return nil
}
