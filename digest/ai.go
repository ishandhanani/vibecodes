package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	openrouter "github.com/revrost/go-openrouter"
)

const batchSize = 15 // Items per AI request

// AIAnalyzer handles AI-powered analysis of digest items
type AIAnalyzer struct {
	client  *openrouter.Client
	model   string
	enabled bool
}

// NewAIAnalyzer creates a new AI analyzer from config
func NewAIAnalyzer(cfg AIConfig) *AIAnalyzer {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		logDebug("AI disabled: OPENROUTER_API_KEY not set")
		return &AIAnalyzer{enabled: false}
	}
	if !cfg.Enabled {
		logDebug("AI disabled: config.ai.enabled = false")
		return &AIAnalyzer{enabled: false}
	}

	logDebug("AI enabled with model: %s", cfg.Model)
	return &AIAnalyzer{
		client:  openrouter.NewClient(apiKey),
		model:   cfg.Model,
		enabled: true,
	}
}

// IsEnabled returns whether AI analysis is available
func (a *AIAnalyzer) IsEnabled() bool {
	return a.enabled
}

// AnalysisResult contains the AI analysis for an item
type AnalysisResult struct {
	Summary  string `json:"summary"`
	Priority int    `json:"priority"`
}

// AnalysisResponse is the structured output schema for AI analysis
type AnalysisResponse struct {
	Items []AnalysisResult `json:"items"`
}

// analysisSchema implements json.Marshaler for the structured output schema
type analysisSchema struct{}

func (s analysisSchema) MarshalJSON() ([]byte, error) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"summary": map[string]any{
							"type":        "string",
							"description": "Brief 1-sentence summary of the item, max 80 characters",
						},
						"priority": map[string]any{
							"type":        "integer",
							"description": "Priority score 1-5 (5=urgent/directly involves user, 1=background noise)",
							"minimum":     1,
							"maximum":     5,
						},
					},
					"required":             []string{"summary", "priority"},
					"additionalProperties": false,
				},
			},
		},
		"required":             []string{"items"},
		"additionalProperties": false,
	}
	return json.Marshal(schema)
}

// AnalyzeItems analyzes items in concurrent batches
func (a *AIAnalyzer) AnalyzeItems(ctx context.Context, items []DigestItem, user string, interests []string, exclusions []string) ([]DigestItem, error) {
	if !a.enabled || len(items) == 0 {
		return items, nil
	}

	// Split into batches
	var batches [][]DigestItem
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batches = append(batches, items[i:end])
	}

	logDebug("Processing %d items in %d batches (size %d)", len(items), len(batches), batchSize)

	// Process batches concurrently
	type batchResult struct {
		index   int
		results []AnalysisResult
		err     error
	}

	resultsChan := make(chan batchResult, len(batches))
	var wg sync.WaitGroup

	for i, batch := range batches {
		wg.Add(1)
		go func(idx int, batchItems []DigestItem) {
			defer wg.Done()
			results, err := a.analyzeBatch(ctx, batchItems, user, interests, exclusions)
			resultsChan <- batchResult{index: idx, results: results, err: err}
		}(i, batch)
	}

	// Wait and close channel
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Collect results
	allResults := make([][]AnalysisResult, len(batches))
	var errs []error
	for result := range resultsChan {
		if result.err != nil {
			errs = append(errs, result.err)
			logDebug("Batch %d failed: %v", result.index, result.err)
		} else {
			allResults[result.index] = result.results
			logDebug("Batch %d completed: %d results", result.index, len(result.results))
		}
	}

	// Apply results to items
	itemIdx := 0
	for batchIdx, batchResults := range allResults {
		batchStart := batchIdx * batchSize
		for i, result := range batchResults {
			itemIdx = batchStart + i
			if itemIdx < len(items) {
				items[itemIdx].Summary = result.Summary
				items[itemIdx].Priority = result.Priority
			}
		}
	}

	if len(errs) > 0 {
		return items, fmt.Errorf("%d/%d batches failed", len(errs), len(batches))
	}
	return items, nil
}

func (a *AIAnalyzer) analyzeBatch(ctx context.Context, items []DigestItem, user string, interests []string, exclusions []string) ([]AnalysisResult, error) {
	// Build item descriptions
	var itemDescriptions []string
	for i, item := range items {
		itemDescriptions = append(itemDescriptions, fmt.Sprintf(
			"%d. [%s] %s - %s",
			i+1, item.Section, item.Title, item.Sub,
		))
	}

	interestsStr := "none specified"
	if len(interests) > 0 {
		interestsStr = join(interests, ", ")
	}

	excludeStr := "none"
	if len(exclusions) > 0 {
		excludeStr = join(exclusions, ", ")
	}

	prompt := fmt.Sprintf(`Analyze these GitHub items for @%s.

INTERESTS (boost priority): %s
EXCLUDE (priority 1): %s

For each item:
- summary: 1-sentence description (max 80 chars)
- priority: 1-5 where:
  5 = User is author OR explicitly @mentioned
  4 = Matches interests AND significant
  3 = Matches interests
  2 = Tangentially related, routine reviewer tag
  1 = Matches EXCLUDE, unrelated, CI/docs

Items:
%s

Return %d items.`, user, interestsStr, excludeStr, join(itemDescriptions, "\n"), len(items))

	req := openrouter.ChatCompletionRequest{
		Model: a.model,
		Messages: []openrouter.ChatCompletionMessage{
			{
				Role:    "user",
				Content: openrouter.Content{Text: prompt},
			},
		},
		ResponseFormat: &openrouter.ChatCompletionResponseFormat{
			Type: openrouter.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openrouter.ChatCompletionResponseFormatJSONSchema{
				Name:        "analysis",
				Description: "Analysis results for GitHub items",
				Schema:      analysisSchema{},
				Strict:      true,
			},
		},
	}

	resp, err := a.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("API error: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty response")
	}

	var analysis AnalysisResponse
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content.Text), &analysis); err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	return analysis.Items, nil
}

func join(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for _, s := range strs[1:] {
		result += sep + s
	}
	return result
}
