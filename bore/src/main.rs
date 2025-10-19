mod app;
mod config;
mod ui;

use app::App;
use clap::Parser;
use config::Config;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "bore")]
#[command(about = "SSH Port Forward TUI", long_about = None)]
struct Cli {
    /// Use a preset configuration
    #[arg(short, long)]
    preset: Option<String>,

    /// Create example config file
    #[arg(long)]
    init_config: bool,

    /// List available presets
    #[arg(long)]
    list_presets: bool,
}

fn get_input(prompt: &str) -> io::Result<String> {
    print!("{}", prompt);
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn main() -> Result<(), io::Error> {
    let cli = Cli::parse();

    // Handle --init-config
    if cli.init_config {
        match Config::create_example_config() {
            Ok(_) => {
                println!("✓ Created example config at: {}", config::get_config_path_display());
                println!("\nEdit this file to add your presets, then run:");
                println!("  bore --preset <name>");
                return Ok(());
            }
            Err(e) => {
                eprintln!("✗ Failed to create config: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Load config
    let config = match Config::load() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Warning: Failed to load config: {}", e);
            eprintln!("Run 'bore --init-config' to create a config file\n");
            Config { presets: std::collections::HashMap::new() }
        }
    };

    // Handle --list-presets
    if cli.list_presets {
        if config.presets.is_empty() {
            println!("No presets found. Run 'bore --init-config' to create a config file.");
        } else {
            println!("Available presets:\n");
            for (name, preset) in &config.presets {
                println!("  {} → {} [{}]",
                    name,
                    preset.host,
                    preset.ports.iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            println!("\nUsage: bore --preset <name>");
        }
        return Ok(());
    }

    // Get host and ports (from preset or interactive)
    let (host, ports) = if let Some(preset_name) = cli.preset {
        match config.get_preset(&preset_name) {
            Some(preset) => {
                println!("Using preset '{}': {} [{}]",
                    preset_name,
                    preset.host,
                    preset.ports.iter()
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                (preset.host.clone(), preset.ports.clone())
            }
            None => {
                eprintln!("✗ Preset '{}' not found", preset_name);
                eprintln!("Run 'bore --list-presets' to see available presets");
                std::process::exit(1);
            }
        }
    } else {
        // Interactive mode
        let host = get_input("SSH host (e.g., 'dev' or 'user@dev'): ")?;
        if host.is_empty() {
            eprintln!("Error: Host cannot be empty");
            std::process::exit(1);
        }

        let ports_input = get_input("Ports to forward (comma-separated, e.g., '3001,8080,4002'): ")?;
        let ports: Vec<u16> = ports_input
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        if ports.is_empty() {
            eprintln!("Error: No valid ports provided");
            std::process::exit(1);
        }

        (host, ports)
    };

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
        terminal.draw(|f| ui::render(f, app))?;

        if app.should_check_health() {
            app.check_port_health();
        }

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => {
                        app.stop_all_tunnels();
                        return Ok(());
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        app.move_selection_up();
                    }
                    KeyCode::Char('j') | KeyCode::Down => {
                        app.move_selection_down();
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        app.toggle_selected_tunnel();
                    }
                    _ => {}
                }
            }
        }
    }
}
