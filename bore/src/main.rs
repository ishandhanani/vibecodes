use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Terminal,
};
use std::io::{self, Write};
use std::process::{Child, Command};

struct App {
    ports: Vec<u16>,
    host: String,
    ssh_process: Option<Child>,
    status: String,
}

impl App {
    fn new(host: String, ports: Vec<u16>) -> App {
        App {
            ports,
            host,
            ssh_process: None,
            status: String::from("Press 's' to start, 'q' to quit"),
        }
    }

    fn start_ssh(&mut self) {
        let mut args = vec!["-N".to_string()];

        for port in &self.ports {
            args.push("-L".to_string());
            args.push(format!("{}:localhost:{}", port, port));
        }

        args.push(self.host.clone());

        match Command::new("ssh").args(&args).spawn() {
            Ok(child) => {
                self.ssh_process = Some(child);
                self.status = String::from("✓ SSH tunnel active");
            }
            Err(e) => {
                self.status = format!("✗ Error: {}", e);
            }
        }
    }

    fn stop_ssh(&mut self) {
        if let Some(mut child) = self.ssh_process.take() {
            let _ = child.kill();
            self.status = String::from("Press 's' to start, 'q' to quit");
        }
    }
}

fn get_input(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn main() -> Result<(), io::Error> {
    // Get SSH host/user
    let host = get_input("SSH host (e.g., 'dev' or 'user@dev'): ")?;
    if host.is_empty() {
        eprintln!("Error: Host cannot be empty");
        std::process::exit(1);
    }

    // Get ports
    let ports_input = get_input("Ports to forward (comma-separated, e.g., '3001,8080,4002'): ")?;
    let ports: Vec<u16> = ports_input
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if ports.is_empty() {
        eprintln!("Error: No valid ports provided");
        std::process::exit(1);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(host, ports);
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> io::Result<()> {
    loop {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(5),
                    Constraint::Length(3),
                ])
                .split(f.area());

            let title = Paragraph::new("SSH Port Forward")
                .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            let items: Vec<ListItem> = vec![
                ListItem::new(Line::from(vec![
                    Span::styled("Host: ", Style::default().fg(Color::Yellow)),
                    Span::raw(&app.host),
                ])),
                ListItem::new(Line::from(vec![
                    Span::styled("Ports: ", Style::default().fg(Color::Yellow)),
                    Span::raw(
                        app.ports
                            .iter()
                            .map(|p| p.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    ),
                ])),
            ];

            let list = List::new(items).block(Block::default().borders(Borders::ALL).title("Config"));
            f.render_widget(list, chunks[1]);

            let status_color = if app.ssh_process.is_some() {
                Color::Green
            } else {
                Color::Gray
            };

            let status = Paragraph::new(app.status.clone())
                .style(Style::default().fg(status_color))
                .block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(status, chunks[2]);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => {
                        app.stop_ssh();
                        return Ok(());
                    }
                    KeyCode::Char('s') => {
                        if app.ssh_process.is_none() {
                            app.start_ssh();
                        }
                    }
                    KeyCode::Char('x') => {
                        app.stop_ssh();
                    }
                    _ => {}
                }
            }
        }
    }
}
