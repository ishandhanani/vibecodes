package main

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// GenerateMarkdown creates a markdown digest from items
func GenerateMarkdown(items []DigestItem, cfg Config) string {
	var sb strings.Builder

	// Deduplicate items by URL
	seen := make(map[string]bool)
	var uniqueItems []DigestItem
	for _, item := range items {
		if !seen[item.URL] {
			seen[item.URL] = true
			uniqueItems = append(uniqueItems, item)
		}
	}
	items = uniqueItems

	// Header
	sb.WriteString("# GitHub Digest\n\n")
	sb.WriteString(fmt.Sprintf("**Generated:** %s  \n", time.Now().Format("Mon Jan 2, 2006 3:04 PM")))
	sb.WriteString(fmt.Sprintf("**Looking back:** %d day(s)  \n", cfg.DaysBack))
	sb.WriteString(fmt.Sprintf("**Items:** %d\n\n", len(items)))

	if len(items) == 0 {
		sb.WriteString("_No activity found._\n")
		return sb.String()
	}

	// Check if AI analysis ran (any item has priority > 0)
	aiRan := false
	for _, item := range items {
		if item.Priority > 0 {
			aiRan = true
			break
		}
	}

	if aiRan {
		// AI mode: organize by priority
		sb.WriteString(generateAIMode(items))
	} else {
		// No AI: organize by repo
		sb.WriteString(generateRepoMode(items))
	}

	return sb.String()
}

func generateAIMode(items []DigestItem) string {
	var sb strings.Builder

	// Sort by priority desc, then time
	sort.Slice(items, func(i, j int) bool {
		if items[i].Priority != items[j].Priority {
			return items[i].Priority > items[j].Priority
		}
		return items[i].When.After(items[j].When)
	})

	// Separate into buckets
	var high, med, low []DigestItem
	for _, item := range items {
		switch {
		case item.Priority >= 4:
			high = append(high, item)
		case item.Priority >= 2:
			med = append(med, item)
		default:
			low = append(low, item)
		}
	}

	sb.WriteString(fmt.Sprintf("**🔥 High:** %d | **📋 Review:** %d | **📁 Low:** %d\n\n",
		len(high), len(med), len(low)))

	// High priority
	if len(high) > 0 {
		sb.WriteString("## 🔥 Action Required\n\n")
		for _, item := range high {
			sb.WriteString(formatFull(item))
		}
		sb.WriteString("\n")
	}

	// Medium priority grouped by repo
	if len(med) > 0 {
		sb.WriteString("## 📋 Worth Reviewing\n\n")
		sb.WriteString(groupByRepo(med, true))
	}

	// Low priority collapsed
	if len(low) > 0 {
		sb.WriteString("## 📁 Low Priority\n\n")
		sb.WriteString("<details>\n<summary>")
		sb.WriteString(fmt.Sprintf("%d items", len(low)))
		sb.WriteString("</summary>\n\n")
		sb.WriteString(groupByRepo(low, false))
		sb.WriteString("</details>\n")
	}

	return sb.String()
}

func generateRepoMode(items []DigestItem) string {
	var sb strings.Builder

	// Group by repo, then by section
	type repoData struct {
		sections map[string][]DigestItem
		order    []string
	}
	repos := make(map[string]*repoData)
	repoOrder := []string{}

	for _, item := range items {
		if repos[item.RepoName] == nil {
			repos[item.RepoName] = &repoData{
				sections: make(map[string][]DigestItem),
			}
			repoOrder = append(repoOrder, item.RepoName)
		}
		rd := repos[item.RepoName]
		if _, exists := rd.sections[item.Section]; !exists {
			rd.order = append(rd.order, item.Section)
		}
		rd.sections[item.Section] = append(rd.sections[item.Section], item)
	}

	for _, repoName := range repoOrder {
		rd := repos[repoName]
		sb.WriteString(fmt.Sprintf("## %s\n\n", repoName))

		for _, secName := range rd.order {
			secItems := rd.sections[secName]

			// Extract section suffix
			displaySec := secName
			if idx := strings.Index(secName, " · "); idx != -1 {
				displaySec = secName[idx+len(" · "):]
			}

			sb.WriteString(fmt.Sprintf("### %s\n\n", displaySec))

			// Sort by time
			sort.Slice(secItems, func(i, j int) bool {
				return secItems[i].When.After(secItems[j].When)
			})

			for _, item := range secItems {
				sb.WriteString(formatCompact(item))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func groupByRepo(items []DigestItem, showPriority bool) string {
	var sb strings.Builder

	byRepo := make(map[string][]DigestItem)
	repoOrder := []string{}
	for _, item := range items {
		if _, exists := byRepo[item.RepoName]; !exists {
			repoOrder = append(repoOrder, item.RepoName)
		}
		byRepo[item.RepoName] = append(byRepo[item.RepoName], item)
	}

	for _, repoName := range repoOrder {
		repoItems := byRepo[repoName]
		sb.WriteString(fmt.Sprintf("### %s\n\n", repoName))
		for _, item := range repoItems {
			if showPriority {
				sb.WriteString(formatCompact(item))
			} else {
				sb.WriteString(formatMinimal(item))
			}
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

func formatFull(item DigestItem) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("- **[P%d]** [%s](%s) _%s_\n", item.Priority, item.Title, item.URL, item.RepoName))
	if item.Summary != "" {
		sb.WriteString(fmt.Sprintf("  > %s\n", item.Summary))
	}
	sb.WriteString(fmt.Sprintf("  %s\n", item.Sub))
	return sb.String()
}

func formatCompact(item DigestItem) string {
	var sb strings.Builder
	if item.Priority > 0 {
		sb.WriteString(fmt.Sprintf("- **[P%d]** ", item.Priority))
	} else {
		sb.WriteString("- ")
	}
	sb.WriteString(fmt.Sprintf("[%s](%s)", item.Title, item.URL))
	if item.Summary != "" {
		sb.WriteString(fmt.Sprintf(" - _%s_", item.Summary))
	}
	sb.WriteString(fmt.Sprintf(" · %s\n", item.Sub))
	return sb.String()
}

func formatMinimal(item DigestItem) string {
	return fmt.Sprintf("- [%s](%s)\n", item.Title, item.URL)
}
