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
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
enum PortStatus {
    Unknown,
    Healthy,
    Unhealthy,
}

struct App {
    ports: Vec<u16>,
    port_status: Vec<PortStatus>,
    host: String,
    ssh_process: Option<Child>,
    status: String,
    last_health_check: Option<Instant>,
}

impl App {
    fn new(host: String, ports: Vec<u16>) -> App {
        let port_count = ports.len();
        App {
            ports,
            port_status: vec![PortStatus::Unknown; port_count],
            host,
            ssh_process: None,
            status: String::from("Press 's' to start, 'q' to quit"),
            last_health_check: None,
        }
    }

    fn check_port_health(&mut self) {
        if self.ssh_process.is_none() {
            // Reset all to unknown if tunnel is down
            for status in &mut self.port_status {
                *status = PortStatus::Unknown;
            }
            return;
        }

        for (idx, port) in self.ports.iter().enumerate() {
            // Try to connect to localhost:port with a very short timeout
            let addr = format!("127.0.0.1:{}", port);
            match TcpStream::connect_timeout(
                &addr.parse().unwrap(),
                Duration::from_millis(100),
            ) {
                Ok(_) => {
                    self.port_status[idx] = PortStatus::Healthy;
                }
                Err(_) => {
                    self.port_status[idx] = PortStatus::Unhealthy;
                }
            }
        }

        self.last_health_check = Some(Instant::now());
    }

    fn should_check_health(&self) -> bool {
        if self.ssh_process.is_none() {
            return false;
        }

        match self.last_health_check {
            None => true,
            Some(last) => last.elapsed() > Duration::from_secs(2),
        }
    }

    fn start_ssh(&mut self) {
        let mut args = vec!["-N".to_string()];

        for port in &self.ports {
            args.push("-L".to_string());
            args.push(format!("{}:localhost:{}", port, port));
        }

        args.push(self.host.clone());

        match Command::new("ssh")
            .args(&args)
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                // Check if process is still running after a brief moment
                std::thread::sleep(std::time::Duration::from_millis(200));

                match child.try_wait() {
                    Ok(Some(status)) => {
                        // Process exited, read stderr for error
                        let stderr = child.stderr.take();
                        if let Some(stderr) = stderr {
                            let reader = BufReader::new(stderr);
                            let errors: Vec<String> = reader.lines()
                                .filter_map(|line| line.ok())
                                .collect();
                            if !errors.is_empty() {
                                self.status = format!("✗ SSH failed: {}", errors.join("; "));
                            } else {
                                self.status = format!("✗ SSH exited with code {:?}", status.code());
                            }
                        } else {
                            self.status = format!("✗ SSH failed with code {:?}", status.code());
                        }
                    }
                    Ok(None) => {
                        // Still running, success!
                        self.ssh_process = Some(child);
                        self.status = String::from("✓ SSH tunnel active");
                    }
                    Err(e) => {
                        self.status = format!("✗ Error checking SSH status: {}", e);
                    }
                }
            }
            Err(e) => {
                self.status = format!("✗ Failed to start SSH: {}", e);
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
                .margin(1)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Min(4),
                    Constraint::Length(3),
                ])
                .split(f.area());

            // Title with futuristic styling
            let title = Paragraph::new("⚡ SSH PORT FORWARD")
                .style(Style::default()
                    .fg(Color::Rgb(0, 255, 255))
                    .add_modifier(Modifier::BOLD))
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(0, 200, 255))));
            f.render_widget(title, chunks[0]);

            // Host info
            let is_active = app.ssh_process.is_some();
            let host_status = if is_active { "●" } else { "○" };
            let host_color = if is_active { Color::Rgb(0, 255, 150) } else { Color::Gray };

            let host_info = Paragraph::new(Line::from(vec![
                Span::styled(host_status, Style::default().fg(host_color).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(&app.host, Style::default().fg(Color::Rgb(150, 150, 255))),
            ]))
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 120)))
                .title("TARGET"));
            f.render_widget(host_info, chunks[1]);

            // Ports list with status indicators
            let port_items: Vec<ListItem> = app.ports
                .iter()
                .enumerate()
                .map(|(idx, port)| {
                    let health_status = app.port_status[idx];

                    let (status_icon, port_color) = match (is_active, health_status) {
                        (false, _) => ("▹", Color::DarkGray),
                        (true, PortStatus::Healthy) => ("●", Color::Rgb(0, 255, 150)),
                        (true, PortStatus::Unhealthy) => ("●", Color::Rgb(255, 50, 50)),
                        (true, PortStatus::Unknown) => ("◐", Color::Rgb(255, 200, 0)),
                    };

                    let text_color = match (is_active, health_status) {
                        (false, _) => Color::Gray,
                        (true, PortStatus::Healthy) => Color::White,
                        (true, PortStatus::Unhealthy) => Color::Rgb(255, 150, 150),
                        (true, PortStatus::Unknown) => Color::Rgb(200, 200, 200),
                    };

                    ListItem::new(Line::from(vec![
                        Span::styled(status_icon, Style::default().fg(port_color)),
                        Span::raw(" "),
                        Span::styled(
                            format!("localhost:{}", port),
                            Style::default().fg(text_color)
                        ),
                        Span::raw(" "),
                        Span::styled(
                            if is_active { "→" } else { "·" },
                            Style::default().fg(Color::DarkGray)
                        ),
                        Span::raw(" "),
                        Span::styled(
                            format!("{}:{}", app.host.split('@').last().unwrap_or(&app.host), port),
                            Style::default().fg(if is_active { Color::Rgb(150, 150, 255) } else { Color::DarkGray })
                        ),
                    ]))
                })
                .collect();

            let ports_list = List::new(port_items)
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(80, 80, 120)))
                    .title("TUNNELS"));
            f.render_widget(ports_list, chunks[2]);

            // Status bar
            let status_color = if app.status.starts_with("✓") {
                Color::Rgb(0, 255, 150)
            } else if app.status.starts_with("✗") {
                Color::Rgb(255, 50, 50)
            } else {
                Color::DarkGray
            };

            let status = Paragraph::new(app.status.clone())
                .style(Style::default().fg(status_color))
                .block(Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Rgb(80, 80, 120))));
            f.render_widget(status, chunks[3]);
        })?;

        // Check port health periodically
        if app.should_check_health() {
            app.check_port_health();
        }

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
