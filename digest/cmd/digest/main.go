package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/lipgloss"
)

// ---------- Config & constants ----------

const (
	defaultUser        = "ishandhanani"
	defaultSGLangRepo  = "sgl-project/sglang"
	defaultDynamoRepo  = "ai-dynamo/dynamo"
	stateFileName      = "state.json"
	defaultDaysBack    = 1
	githubAPIBase      = "https://api.github.com"
	searchIssuesPath   = "/search/issues"
	sglangPullsPathTpl = "/repos/%s/pulls"
)

type CLIFlags struct {
	Since       time.Time
	WindowDays  int
	User        string
	SGLangRepo  string
	DynamoRepo  string
}

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

func loadState() *State {
	b, err := os.ReadFile(filepath.Join(stateDir(), stateFileName))
	if err != nil {
		return &State{}
	}
	var s State
	if json.Unmarshal(b, &s) != nil {
		return &State{}
	}
	return &State{}
}

func saveState(now time.Time) {
	_ = os.MkdirAll(stateDir(), 0o755)
	s := State{LastRunISO: now.UTC().Format(time.RFC3339)}
	_ = os.WriteFile(filepath.Join(stateDir(), stateFileName), mustJSON(s), 0o644)
}

func mustJSON(v any) []byte {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil { panic(err) }
	return b
}

// ---------- GitHub API ----------

func ghToken() (string, error) {
	t := os.Getenv("GITHUB_TOKEN")
	if t == "" {
		return "", errors.New("GITHUB_TOKEN is not set")
	}
	return t, nil
}

func httpClient() *http.Client {
	return &http.Client{
		Timeout: 15 * time.Second,
	}
}

func ghGET(ctx context.Context, path string, q url.Values) (*http.Response, error) {
	tok, err := ghToken()
	if err != nil { return nil, err }
	u := githubAPIBase + path
	if len(q) > 0 {
		u += "?" + q.Encode()
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", u, nil)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+tok)
	return httpClient().Do(req)
}

// minimal structs

type SearchResult struct {
	TotalCount int           `json:"total_count"`
	Items      []IssueOrPR   `json:"items"`
}

type IssueOrPR struct {
	HTMLURL   string    `json:"html_url"`
	Title     string    `json:"title"`
	Number    int       `json:"number"`
	UpdatedAt time.Time `json:"updated_at"`
	CreatedAt time.Time `json:"created_at"`
	State     string    `json:"state"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
	PullRequest *struct {
		URL string `json:"url"`
	} `json:"pull_request"`
}

type Pull struct {
	HTMLURL   string    `json:"html_url"`
	Title     string    `json:"title"`
	Number    int       `json:"number"`
	CreatedAt time.Time `json:"created_at"`
	User      struct {
		Login string `json:"login"`
	} `json:"user"`
}

func searchIssues(ctx context.Context, query string, perPage int) ([]IssueOrPR, error) {
	q := url.Values{}
	q.Set("q", query)
	q.Set("per_page", fmt.Sprintf("%d", perPage))
	resp, err := ghGET(ctx, searchIssuesPath, q)
	if err != nil { return nil, err }
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GitHub search failed: %s", resp.Status)
	}
	var sr SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, err
	}
	return sr.Items, nil
}

func listNewPRsSince(ctx context.Context, repo string, since time.Time, perPage int) ([]Pull, error) {
	path := fmt.Sprintf(sglangPullsPathTpl, repo)
	q := url.Values{}
	q.Set("state", "open")
	q.Set("sort", "created")
	q.Set("direction", "desc")
	q.Set("per_page", fmt.Sprintf("%d", perPage))
	resp, err := ghGET(ctx, "/repos/"+repo+"/pulls", q)
	if err != nil { return nil, err }
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

// queries we need:
//
// 1) sglang mentions of @ishandhanani since <since>:
//    q = repo:sgl-project/sglang mentions:ishandhanani updated:>=YYYY-MM-DD
//
// 2) sglang new PRs since <since> (via pulls API)
//
// 3) dynamo issues referencing sglang since <since>:
//    q = repo:ai-dynamo/dynamo "sglang" is:issue updated:>=YYYY-MM-DD

type DigestItem struct {
	Section string // "SGLang Mentions" / "SGLang New PRs" / "Dynamo Issues (sglang)"
	Title   string
	Sub     string
	URL     string
	When    time.Time
}

func collect(ctx context.Context, cfg CLIFlags) ([]DigestItem, error) {
	sinceDate := cfg.Since
	if sinceDate.IsZero() {
		sinceDate = time.Now().Add(-time.Duration(cfg.WindowDays) * 24 * time.Hour)
	}
	yyyyMmDd := sinceDate.UTC().Format("2006-01-02")

	var items []DigestItem

	// 1) SGLang mentions
	mentionQ := fmt.Sprintf(`repo:%s mentions:%s updated:>=%s`, cfg.SGLangRepo, cfg.User, yyyyMmDd)
	mentionResults, err := searchIssues(ctx, mentionQ, 50)
	if err != nil { return nil, err }
	for _, it := range mentionResults {
		items = append(items, DigestItem{
			Section: "SGLang · Mentions",
			Title:   fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:     fmt.Sprintf("updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:     it.HTMLURL,
			When:    it.UpdatedAt,
		})
	}

	// 2) SGLang new PRs since
	prs, err := listNewPRsSince(ctx, cfg.SGLangRepo, sinceDate, 50)
	if err != nil { return nil, err }
	for _, pr := range prs {
		items = append(items, DigestItem{
			Section: "SGLang · New PRs",
			Title:   fmt.Sprintf("#%d %s", pr.Number, pr.Title),
			Sub:     fmt.Sprintf("opened %s by @%s", humanWhen(pr.CreatedAt), pr.User.Login),
			URL:     pr.HTMLURL,
			When:    pr.CreatedAt,
		})
	}

	// 3) Dynamo issues referencing sglang
	dynQ := fmt.Sprintf(`repo:%s "sglang" is:issue updated:>=%s`, cfg.DynamoRepo, yyyyMmDd)
	dynResults, err := searchIssues(ctx, dynQ, 50)
	if err != nil { return nil, err }
	for _, it := range dynResults {
		items = append(items, DigestItem{
			Section: "Dynamo · Issues mentioning “sglang”",
			Title:   fmt.Sprintf("#%d %s", it.Number, it.Title),
			Sub:     fmt.Sprintf("updated %s by @%s", humanWhen(it.UpdatedAt), it.User.Login),
			URL:     it.HTMLURL,
			When:    it.UpdatedAt,
		})
	}

	sort.Slice(items, func(i, j int) bool {
		if items[i].Section == items[j].Section {
			return items[i].When.After(items[j].When)
		}
		return items[i].Section < items[j].Section
	})

	return items, nil
}

// ---------- TUI ----------

type listItem struct {
	title, desc, url, section string
	when                      time.Time
}

func (i listItem) Title() string       { return i.title }
func (i listItem) Description() string { return i.desc }
func (i listItem) FilterValue() string { return i.title + " " + i.desc + " " + i.section }

type model struct {
	list     list.Model
	status   string
	cfg      CLIFlags
}

var (
	titleStyle   = lipgloss.NewStyle().Bold(true)
	sectionStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("69"))
	statusStyle  = lipgloss.NewStyle().Faint(true)
)

func newModel(cfg CLIFlags, items []DigestItem) model {
	l := list.New(convert(items), list.NewDefaultDelegate(), 0, 0)
	l.Title = titleStyle.Render("digest")
	l.SetShowHelp(true)
	l.SetFilteringEnabled(true)
	l.SetShowStatusBar(false)
	l.SetShowPagination(true)
	l.Styles.Title = lipgloss.NewStyle().Bold(true)
	return model{list: l, cfg: cfg, status: "r: refresh · o: open · q: quit"}
}

func convert(ds []DigestItem) []list.Item {
	out := make([]list.Item, 0, len(ds))
	for _, d := range ds {
		out = append(out, listItem{
			title:   fmt.Sprintf("%s  %s", sectionStyle.Render("["+d.Section+"]"), d.Title),
			desc:    d.Sub,
			url:     d.URL,
			section: d.Section,
			when:    d.When,
		})
	}
	return out
}

type refreshMsg struct {
	items []DigestItem
	err   error
}

func (m model) Init() tea.Cmd {
	return nil
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.list.SetSize(msg.Width, msg.Height-2)
	case tea.KeyMsg:
		switch msg.String() {
		case "q", "esc", "ctrl+c":
			return m, tea.Quit
		case "r":
			m.status = "Refreshing…"
			return m, fetchCmd(m.cfg)
		case "o":
			if it, ok := m.list.SelectedItem().(listItem); ok {
				_ = openBrowser(it.url)
				m.status = "Opened in browser"
			}
		}
	case refreshMsg:
		if msg.err != nil {
			m.status = "Error: " + msg.err.Error()
			return m, nil
		}
		m.list.SetItems(convert(msg.items))
		m.status = fmt.Sprintf("Updated %s", humanWhen(time.Now()))
		saveState(time.Now())
	}
	var cmd tea.Cmd
	m.list, cmd = m.list.Update(msg)
	return m, cmd
}

func (m model) View() string {
	return fmt.Sprintf("%s\n%s", m.list.View(), statusStyle.Render(m.status))
}

func fetchCmd(cfg CLIFlags) tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		items, err := collect(ctx, cfg)
		return refreshMsg{items: items, err: err}
	}
}

func openBrowser(u string) error {
	switch runtime.GOOS {
	case "darwin":
		return exec.Command("open", u).Start()
	case "linux":
		return exec.Command("xdg-open", u).Start()
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", u).Start()
	default:
		return fmt.Errorf("unsupported OS")
	}
}

// ---------- Utilities & entry ----------

func humanWhen(t time.Time) string {
	if t.IsZero() { return "never" }
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

func parseFlags() CLIFlags {
	// ultra-simple flag parsing (no external deps)
	cfg := CLIFlags{
		WindowDays: defaultDaysBack,
		User:       getenv("DIGEST_USER", defaultUser),
		SGLangRepo: getenv("DIGEST_SGLANG_REPO", defaultSGLangRepo),
		DynamoRepo: getenv("DIGEST_DYNAMO_REPO", defaultDynamoRepo),
	}
	args := os.Args[1:]
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--since":
			if i+1 < len(args) {
				if ts, err := time.Parse("2006-01-02", args[i+1]); err == nil {
					cfg.Since = ts
					i++
				}
			}
		case "--days":
			if i+1 < len(args) {
				if n, err := parseInt(args[i+1]); err == nil && n > 0 {
					cfg.WindowDays = n
					i++
				}
			}
		case "--user":
			if i+1 < len(args) { cfg.User = args[i+1]; i++ }
		case "--sglang-repo":
			if i+1 < len(args) { cfg.SGLangRepo = args[i+1]; i++ }
		case "--dynamo-repo":
			if i+1 < len(args) { cfg.DynamoRepo = args[i+1]; i++ }
		}
	}
	return cfg
}

func parseInt(s string) (int, error) {
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	return n, err
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" { return v }
	return def
}

func main() {
	cfg := parseFlags()

	// initial load
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	items, err := collect(ctx, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	saveState(time.Now())

	p := tea.NewProgram(newModel(cfg, items), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "tui error: %v\n", err)
		os.Exit(1)
	}
}

