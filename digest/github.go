package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"time"
)

const (
	githubAPIBase    = "https://api.github.com"
	searchIssuesPath = "/search/issues"
)

// DigestItem represents a single item in the digest
type DigestItem struct {
	Section  string
	Title    string
	Sub      string
	URL      string
	When     time.Time
	RepoName string
	Priority int    // 1-5, set by AI analysis
	Summary  string // AI-generated summary
}

// GitHub API response types
type SearchResult struct {
	TotalCount int         `json:"total_count"`
	Items      []IssueOrPR `json:"items"`
}

type IssueOrPR struct {
	HTMLURL   string    `json:"html_url"`
	Title     string    `json:"title"`
	Number    int       `json:"number"`
	UpdatedAt time.Time `json:"updated_at"`
	CreatedAt time.Time `json:"created_at"`
	State     string    `json:"state"`
	Body      string    `json:"body"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
	PullRequest *struct {
		URL string `json:"url"`
	} `json:"pull_request"`
}

type Pull struct {
	HTMLURL   string    `json:"html_url"`
	Title     string    `json:"title"`
	Number    int       `json:"number"`
	CreatedAt time.Time `json:"created_at"`
	Body      string    `json:"body"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
}

func ghToken() (string, error) {
	t := os.Getenv("GITHUB_TOKEN")
	if t == "" {
		return "", errors.New("GITHUB_TOKEN is not set")
	}
	return t, nil
}

func httpClient() *http.Client {
	return &http.Client{Timeout: 15 * time.Second}
}

func ghGET(ctx context.Context, path string, q url.Values) (*http.Response, error) {
	tok, err := ghToken()
	if err != nil {
		return nil, err
	}
	u := githubAPIBase + path
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+tok)
	return httpClient().Do(req)
}

func searchIssues(ctx context.Context, query string, perPage int) ([]IssueOrPR, error) {
	q := url.Values{}
	q.Set("q", query)
	q.Set("per_page", fmt.Sprintf("%d", perPage))

	logDebug("Search query: %s", query)

	resp, err := ghGET(ctx, searchIssuesPath, q)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		logDebug("GitHub error response: %s", string(body))
		return nil, fmt.Errorf("GitHub search failed: %s", resp.Status)
	}
	var sr SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, err
	}
	return sr.Items, nil
}

func listPRs(ctx context.Context, repo string, since time.Time, perPage int) ([]Pull, error) {
	q := url.Values{}
	q.Set("state", "open")
	q.Set("sort", "created")
	q.Set("direction", "desc")
	q.Set("per_page", fmt.Sprintf("%d", perPage))
	resp, err := ghGET(ctx, "/repos/"+repo+"/pulls", q)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GitHub pulls failed: %s", resp.Status)
	}
	var prs []Pull
	if err := json.NewDecoder(resp.Body).Decode(&prs); err != nil {
		return nil, err
	}
	out := prs[:0]
	for _, p := range prs {
		if p.CreatedAt.After(since) {
			out = append(out, p)
		}
	}
	return out, nil
}

// FetchWatches fetches all items for a repo's watches
func FetchWatches(ctx context.Context, repo RepoConfig, user string, since time.Time) ([]DigestItem, error) {
	var items []DigestItem
	dateFmt := since.UTC().Format("2006-01-02")

	for _, watch := range repo.Watches {
		switch watch.Type {
		case WatchMentions:
			results, err := fetchMentions(ctx, repo, user, dateFmt)
			if err != nil {
				return nil, fmt.Errorf("%s mentions: %w", repo.Name, err)
			}
			items = append(items, results...)

		case WatchNewPRs:
			results, err := fetchNewPRs(ctx, repo, since)
			if err != nil {
				return nil, fmt.Errorf("%s new_prs: %w", repo.Name, err)
			}
			items = append(items, results...)

		case WatchIssues:
			results, err := fetchIssues(ctx, repo, watch.Keywords, dateFmt)
			if err != nil {
				return nil, fmt.Errorf("%s issues: %w", repo.Name, err)
			}
			items = append(items, results...)

		case WatchLabeled:
			results, err := fetchLabeled(ctx, repo, watch.Label, dateFmt)
			if err != nil {
				return nil, fmt.Errorf("%s labeled: %w", repo.Name, err)
			}
			items = append(items, results...)

		case WatchTopics:
			results, err := fetchTopics(ctx, repo, dateFmt)
			if err != nil {
				return nil, fmt.Errorf("%s topics: %w", repo.Name, err)
			}
			items = append(items, results...)
		}
	}

	return items, nil
}

func fetchMentions(ctx context.Context, repo RepoConfig, user, dateFmt string) ([]DigestItem, error) {
	var items []DigestItem

	// Search open issues involving user
	issueQuery := fmt.Sprintf(`repo:%s involves:%s updated:>=%s is:issue is:open`, repo.Repo, user, dateFmt)
	issueResults, err := searchIssues(ctx, issueQuery, 50)
	if err != nil {
		return nil, err
	}
	for _, it := range issueResults {
		items = append(items, DigestItem{
			Section:  fmt.Sprintf("%s · Mentions", repo.Name),
			Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:      fmt.Sprintf("issue updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:      it.HTMLURL,
			When:     it.UpdatedAt,
			RepoName: repo.Name,
		})
	}

	// Search open PRs involving user
	prQuery := fmt.Sprintf(`repo:%s involves:%s updated:>=%s is:pr is:open`, repo.Repo, user, dateFmt)
	prResults, err := searchIssues(ctx, prQuery, 50)
	if err != nil {
		return nil, err
	}
	for _, it := range prResults {
		items = append(items, DigestItem{
			Section:  fmt.Sprintf("%s · Mentions", repo.Name),
			Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:      fmt.Sprintf("PR updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:      it.HTMLURL,
			When:     it.UpdatedAt,
			RepoName: repo.Name,
		})
	}

	return items, nil
}

func fetchNewPRs(ctx context.Context, repo RepoConfig, since time.Time) ([]DigestItem, error) {
	prs, err := listPRs(ctx, repo.Repo, since, 50)
	if err != nil {
		return nil, err
	}

	var items []DigestItem
	for _, pr := range prs {
		items = append(items, DigestItem{
			Section:  fmt.Sprintf("%s · New PRs", repo.Name),
			Title:    fmt.Sprintf("#%d %s", pr.Number, pr.Title),
			Sub:      fmt.Sprintf("opened %s by @%s", humanWhen(pr.CreatedAt), pr.User.Login),
			URL:      pr.HTMLURL,
			When:     pr.CreatedAt,
			RepoName: repo.Name,
		})
	}
	return items, nil
}

func fetchIssues(ctx context.Context, repo RepoConfig, keywords []string, dateFmt string) ([]DigestItem, error) {
	var items []DigestItem

	// Build query - if keywords provided, search for each
	if len(keywords) == 0 {
		// All open issues
		query := fmt.Sprintf(`repo:%s is:issue is:open updated:>=%s`, repo.Repo, dateFmt)
		results, err := searchIssues(ctx, query, 50)
		if err != nil {
			return nil, err
		}
		for _, it := range results {
			items = append(items, issueToDigestItem(repo, it, ""))
		}
	} else {
		// Open issues with keywords
		for _, kw := range keywords {
			query := fmt.Sprintf(`repo:%s "%s" is:issue is:open updated:>=%s`, repo.Repo, kw, dateFmt)
			results, err := searchIssues(ctx, query, 50)
			if err != nil {
				return nil, err
			}
			for _, it := range results {
				items = append(items, issueToDigestItem(repo, it, kw))
			}
		}
	}

	return items, nil
}

func fetchLabeled(ctx context.Context, repo RepoConfig, label, dateFmt string) ([]DigestItem, error) {
	var items []DigestItem

	// Search open issues with label
	issueQuery := fmt.Sprintf(`repo:%s label:"%s" updated:>=%s is:issue is:open`, repo.Repo, label, dateFmt)
	issueResults, err := searchIssues(ctx, issueQuery, 50)
	if err != nil {
		return nil, err
	}
	for _, it := range issueResults {
		items = append(items, DigestItem{
			Section:  fmt.Sprintf("%s · %s", repo.Name, label),
			Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:      fmt.Sprintf("Issue updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:      it.HTMLURL,
			When:     it.UpdatedAt,
			RepoName: repo.Name,
		})
	}

	// Search open PRs with label
	prQuery := fmt.Sprintf(`repo:%s label:"%s" updated:>=%s is:pr is:open`, repo.Repo, label, dateFmt)
	prResults, err := searchIssues(ctx, prQuery, 50)
	if err != nil {
		return nil, err
	}
	for _, it := range prResults {
		items = append(items, DigestItem{
			Section:  fmt.Sprintf("%s · %s", repo.Name, label),
			Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:      fmt.Sprintf("PR updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:      it.HTMLURL,
			When:     it.UpdatedAt,
			RepoName: repo.Name,
		})
	}

	return items, nil
}

func fetchTopics(ctx context.Context, repo RepoConfig, dateFmt string) ([]DigestItem, error) {
	if len(repo.Interests) == 0 {
		return nil, nil
	}

	var items []DigestItem
	seen := make(map[int]bool) // dedupe by issue number

	// Search for each interest topic (only open items)
	for _, topic := range repo.Interests {
		// Search open issues
		issueQuery := fmt.Sprintf(`repo:%s "%s" updated:>=%s is:issue is:open`, repo.Repo, topic, dateFmt)
		issueResults, err := searchIssues(ctx, issueQuery, 20)
		if err != nil {
			logDebug("Topic search failed for %q: %v", topic, err)
			continue
		}
		for _, it := range issueResults {
			if seen[it.Number] {
				continue
			}
			seen[it.Number] = true
			items = append(items, DigestItem{
				Section:  fmt.Sprintf("%s · %s", repo.Name, topic),
				Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
				Sub:      fmt.Sprintf("issue updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
				URL:      it.HTMLURL,
				When:     it.UpdatedAt,
				RepoName: repo.Name,
			})
		}

		// Search open PRs
		prQuery := fmt.Sprintf(`repo:%s "%s" updated:>=%s is:pr is:open`, repo.Repo, topic, dateFmt)
		prResults, err := searchIssues(ctx, prQuery, 20)
		if err != nil {
			continue
		}
		for _, it := range prResults {
			if seen[it.Number] {
				continue
			}
			seen[it.Number] = true
			items = append(items, DigestItem{
				Section:  fmt.Sprintf("%s · %s", repo.Name, topic),
				Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
				Sub:      fmt.Sprintf("PR updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
				URL:      it.HTMLURL,
				When:     it.UpdatedAt,
				RepoName: repo.Name,
			})
		}
	}

	return items, nil
}

func issueToDigestItem(repo RepoConfig, it IssueOrPR, keyword string) DigestItem {
	section := fmt.Sprintf("%s · Issues", repo.Name)
	if keyword != "" {
		section = fmt.Sprintf("%s · Issues (%s)", repo.Name, keyword)
	}
	return DigestItem{
		Section:  section,
		Title:    fmt.Sprintf("#%d %s", it.Number, it.Title),
		Sub:      fmt.Sprintf("updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
		URL:      it.HTMLURL,
		When:     it.UpdatedAt,
		RepoName: repo.Name,
	}
}

// humanWhen formats a time as a human-readable relative string
func humanWhen(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t)
	if d < time.Minute {
		return "just now"
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	}
	if d < 24*time.Hour {
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	}
	return t.Format("Jan 2, 15:04")
}
