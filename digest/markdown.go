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
	sb.WriteString(fmt.Sprintf("**Looking back:** %d day(s) · **Items:** %d\n\n", cfg.DaysBack, len(items)))

	if len(items) == 0 {
		sb.WriteString("_No activity found._\n")
		return sb.String()
	}

	// Group by repo, then by topic
	type topicData struct {
		issues []DigestItem
		prs    []DigestItem
	}
	type repoData struct {
		topics     map[string]*topicData
		topicOrder []string
		other      []DigestItem // items without topics (from watches)
	}
	repos := make(map[string]*repoData)
	repoOrder := []string{}

	for _, item := range items {
		if repos[item.RepoName] == nil {
			repos[item.RepoName] = &repoData{
				topics: make(map[string]*topicData),
			}
			repoOrder = append(repoOrder, item.RepoName)
		}
		rd := repos[item.RepoName]

		if item.Topic != "" {
			if rd.topics[item.Topic] == nil {
				rd.topics[item.Topic] = &topicData{}
				rd.topicOrder = append(rd.topicOrder, item.Topic)
			}
			td := rd.topics[item.Topic]
			if item.ItemType == "issue" {
				td.issues = append(td.issues, item)
			} else {
				td.prs = append(td.prs, item)
			}
		} else {
			rd.other = append(rd.other, item)
		}
	}

	// Render each repo
	for _, repoName := range repoOrder {
		rd := repos[repoName]
		sb.WriteString(fmt.Sprintf("## %s\n\n", repoName))

		// Render topics
		for _, topic := range rd.topicOrder {
			td := rd.topics[topic]
			sb.WriteString(fmt.Sprintf("### %s\n\n", topic))

			// Issues
			if len(td.issues) > 0 {
				sb.WriteString("**Issues**\n\n")
				sortByMention(td.issues)
				for _, item := range td.issues {
					sb.WriteString(formatTopicItem(item))
				}
				sb.WriteString("\n")
			}

			// PRs
			if len(td.prs) > 0 {
				sb.WriteString("**PRs**\n\n")
				sortByMention(td.prs)
				for _, item := range td.prs {
					sb.WriteString(formatTopicItem(item))
				}
				sb.WriteString("\n")
			}
		}

		// Other items (from watches like labeled)
		if len(rd.other) > 0 {
			sb.WriteString("### Other\n\n")
			for _, item := range rd.other {
				sb.WriteString(formatSimple(item))
			}
			sb.WriteString("\n")
		}
	}

	return sb.String()
}

func sortByMention(items []DigestItem) {
	sort.Slice(items, func(i, j int) bool {
		// Mentioned items first
		if items[i].UserMention != items[j].UserMention {
			return items[i].UserMention
		}
		// Then by priority
		if items[i].Priority != items[j].Priority {
			return items[i].Priority > items[j].Priority
		}
		// Then by time
		return items[i].When.After(items[j].When)
	})
}

func formatTopicItem(item DigestItem) string {
	var sb strings.Builder

	// Mention indicator
	if item.UserMention {
		sb.WriteString("- 🔔 ")
	} else {
		sb.WriteString("- ")
	}

	// Priority if set
	if item.Priority > 0 {
		sb.WriteString(fmt.Sprintf("**[P%d]** ", item.Priority))
	}

	// Title with link
	sb.WriteString(fmt.Sprintf("[%s](%s)", item.Title, item.URL))

	// Summary or sub
	if item.Summary != "" {
		sb.WriteString(fmt.Sprintf(" - _%s_", item.Summary))
	}
	sb.WriteString(fmt.Sprintf(" · %s", item.Sub))

	// Mention badge
	if item.UserMention {
		sb.WriteString(" **← you're involved!**")
	}

	sb.WriteString("\n")
	return sb.String()
}

func formatSimple(item DigestItem) string {
	var sb strings.Builder
	if item.UserMention {
		sb.WriteString("- 🔔 ")
	} else {
		sb.WriteString("- ")
	}
	sb.WriteString(fmt.Sprintf("[%s](%s) · %s\n", item.Title, item.URL, item.Sub))
	return sb.String()
}
