use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::app::{App, PortStatus};

pub fn render(f: &mut Frame, app: &App) {
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

    render_title(f, chunks[0]);
    render_host(f, app, chunks[1]);
    render_ports(f, app, chunks[2]);
    render_status(f, app, chunks[3]);
}

fn render_title(f: &mut Frame, area: ratatui::layout::Rect) {
    let title = Paragraph::new("⚡ SSH PORT FORWARD")
        .style(Style::default()
            .fg(Color::Rgb(0, 255, 255))
            .add_modifier(Modifier::BOLD))
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(0, 200, 255))));
    f.render_widget(title, area);
}

fn render_host(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let active_tunnels = app.ssh_processes.iter().filter(|p| p.is_some()).count();
    let host_status = if active_tunnels > 0 { "●" } else { "○" };
    let host_color = if active_tunnels > 0 { Color::Rgb(0, 255, 150) } else { Color::Gray };

    let host_info = Paragraph::new(Line::from(vec![
        Span::styled(host_status, Style::default().fg(host_color).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
        Span::styled(&app.host, Style::default().fg(Color::Rgb(150, 150, 255))),
        Span::raw(" "),
        Span::styled(
            format!("[{}/{}]", active_tunnels, app.ports.len()),
            Style::default().fg(Color::DarkGray)
        ),
    ]))
    .block(Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(80, 80, 120)))
        .title("TARGET"));
    f.render_widget(host_info, area);
}

fn render_ports(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let port_items: Vec<ListItem> = app.ports
        .iter()
        .enumerate()
        .map(|(idx, port)| {
            let is_tunnel_active = app.ssh_processes[idx].is_some();
            let health_status = app.port_status[idx];
            let is_selected = idx == app.selected_index;

            let (status_icon, port_color) = match (is_tunnel_active, health_status) {
                (false, _) => ("▹", Color::DarkGray),
                (true, PortStatus::Healthy) => ("▸", Color::Rgb(0, 255, 150)),
                (true, PortStatus::Unhealthy) => ("▸", Color::Rgb(255, 50, 50)),
                (true, PortStatus::Unknown) => ("◐", Color::Rgb(255, 200, 0)),
            };

            let text_color = match (is_tunnel_active, health_status) {
                (false, _) => Color::Gray,
                (true, PortStatus::Healthy) => Color::White,
                (true, PortStatus::Unhealthy) => Color::Rgb(255, 150, 150),
                (true, PortStatus::Unknown) => Color::Rgb(200, 200, 200),
            };

            let line_spans = vec![
                Span::styled(
                    if is_selected { "▶ " } else { "  " },
                    Style::default().fg(Color::Rgb(0, 255, 255)).add_modifier(Modifier::BOLD)
                ),
                Span::styled(status_icon, Style::default().fg(port_color)),
                Span::raw(" "),
                Span::styled(
                    format!("localhost:{}", port),
                    Style::default().fg(text_color)
                ),
                Span::raw(" "),
                Span::styled(
                    if is_tunnel_active { "→" } else { "·" },
                    Style::default().fg(Color::DarkGray)
                ),
                Span::raw(" "),
                Span::styled(
                    format!("{}:{}", app.host.split('@').last().unwrap_or(&app.host), port),
                    Style::default().fg(if is_tunnel_active { Color::Rgb(150, 150, 255) } else { Color::DarkGray })
                ),
            ];

            let mut item = ListItem::new(Line::from(line_spans));

            if is_selected {
                item = item.style(Style::default().bg(Color::Rgb(20, 40, 60)));
            }

            item
        })
        .collect();

    let ports_list = List::new(port_items)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(80, 80, 120)))
            .title("TUNNELS"));
    f.render_widget(ports_list, area);
}

fn render_status(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
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
    f.render_widget(status, area);
}
