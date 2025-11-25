package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ---------- State persistence ----------

const stateFileName = "state.json"

type State struct {
	LastRunISO string `json:"last_run_iso"`
}

func stateDir() string {
	if d := os.Getenv("DIGEST_STATE_DIR"); d != "" {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "digest")
}

func saveState(now time.Time) {
	_ = os.MkdirAll(stateDir(), 0o755)
	s := State{LastRunISO: now.UTC().Format(time.RFC3339)}
	b, _ := json.MarshalIndent(s, "", "  ")
	_ = os.WriteFile(filepath.Join(stateDir(), stateFileName), b, 0o644)
}

// ---------- Data collection ----------

func collectAll(ctx context.Context, cfg Config) ([]DigestItem, error) {
	since := time.Now().Add(-time.Duration(cfg.DaysBack) * 24 * time.Hour)
	var allItems []DigestItem

	for _, repo := range cfg.Repos {
		items, err := FetchWatches(ctx, repo, cfg.User, since)
		if err != nil {
			return nil, err
		}
		allItems = append(allItems, items...)
	}

	// Sort by section, then by time (newest first)
	sort.Slice(allItems, func(i, j int) bool {
		if allItems[i].Section == allItems[j].Section {
			return allItems[i].When.After(allItems[j].When)
		}
		return allItems[i].Section < allItems[j].Section
	})

	return allItems, nil
}

// ---------- CLI ----------

type cliOptions struct {
	output     string // file path for markdown output, empty for stdout
	configPath string // custom config file path
}

func parseFlags() (cliOptions, *Config) {
	opts := cliOptions{}
	args := os.Args[1:]

	// First pass: look for --config
	for i := 0; i < len(args); i++ {
		if (args[i] == "--config" || args[i] == "-c") && i+1 < len(args) {
			opts.configPath = args[i+1]
			i++
		}
	}

	// Load config
	var cfg Config
	var err error
	if opts.configPath != "" {
		cfg, err = LoadConfigFrom(opts.configPath)
	} else {
		cfg, err = LoadConfig()
	}
	if err != nil {
		logError("config: %v", err)
		os.Exit(1)
	}

	// Second pass: override with other flags
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--days":
			if i+1 < len(args) {
				var n int
				if _, err := fmt.Sscanf(args[i+1], "%d", &n); err == nil && n > 0 {
					cfg.DaysBack = n
					i++
				}
			}
		case "--user":
			if i+1 < len(args) {
				cfg.User = args[i+1]
				i++
			}
		case "--no-ai":
			cfg.AI.Enabled = false
		case "-o", "--output":
			if i+1 < len(args) {
				opts.output = args[i+1]
				i++
			}
		case "-c", "--config":
			i++ // already handled
		case "--help", "-h":
			fmt.Println(`digest - GitHub activity digest with AI analysis

Usage: digest [flags]

Flags:
  -c, --config  Path to config file (default: ~/.config/digest/config.yaml)
  --days N      Look back N days (default: from config or 1)
  --user NAME   GitHub username to track mentions for
  --no-ai       Disable AI analysis
  -o, --output  Write markdown to file instead of stdout

Environment variables:
  GITHUB_TOKEN        Required - GitHub API token
  OPENROUTER_API_KEY  Optional - OpenRouter API key for AI analysis
  DIGEST_LOG_LEVEL    Log level: debug, info, warn, error, silent (default: info)

Config file: ~/.config/digest/config.yaml`)
			os.Exit(0)
		}
	}
	return opts, &cfg
}

func main() {
	// Parse flags and load config
	opts, cfg := parseFlags()

	// Validate
	if err := cfg.Validate(); err != nil {
		logError("config validation: %v", err)
		os.Exit(1)
	}

	// Initialize AI analyzer
	ai := NewAIAnalyzer(cfg.AI)
	logDebug("AI enabled in config: %v, AI analyzer enabled: %v", cfg.AI.Enabled, ai.IsEnabled())

	// Load data
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	logInfo("Loading digest...")
	items, err := collectAll(ctx, *cfg)
	if err != nil {
		logError("%v", err)
		os.Exit(1)
	}

	// Run AI analysis
	if ai.IsEnabled() && len(items) > 0 {
		logInfo("Running AI analysis...")
		items, err = ai.AnalyzeItems(ctx, items, cfg.User, cfg.AllInterests(), cfg.AllExclusions())
		if err != nil {
			logWarn("AI analysis: %v", err)
		}
	}

	saveState(time.Now())

	// Generate markdown
	md := GenerateMarkdown(items, *cfg)
	if opts.output != "" {
		if err := os.WriteFile(opts.output, []byte(md), 0o644); err != nil {
			logError("write: %v", err)
			os.Exit(1)
		}
		logInfo("Written to %s", opts.output)
	} else {
		fmt.Print(md)
	}
}
