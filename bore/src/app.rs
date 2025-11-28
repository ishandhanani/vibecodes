use crate::config::{PortMapping, TunnelType};
use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq)]
pub enum PortStatus {
    Unknown,
    Healthy,
    Unhealthy,
}

pub struct App {
    pub ports: Vec<PortMapping>,
    pub port_status: Vec<PortStatus>,
    pub tunnel_processes: Vec<Option<Child>>,
    pub tunnel_type: TunnelType,
    pub status: String,
    pub selected_index: usize,
    last_health_check: Option<Instant>,
}

impl App {
    pub fn new(tunnel_type: TunnelType, ports: Vec<PortMapping>) -> App {
        let port_count = ports.len();
        let tunnel_processes = (0..port_count).map(|_| None).collect();
        App {
            ports,
            port_status: vec![PortStatus::Unknown; port_count],
            tunnel_processes,
            tunnel_type,
            status: String::from("j/k navigate • Enter toggle • q quit"),
            selected_index: 0,
            last_health_check: None,
        }
    }

    pub fn display_name(&self) -> String {
        self.tunnel_type.display_name()
    }

    pub fn type_label(&self) -> &'static str {
        self.tunnel_type.type_label()
    }

    pub fn move_selection_up(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    pub fn move_selection_down(&mut self) {
        if self.selected_index < self.ports.len().saturating_sub(1) {
            self.selected_index += 1;
        }
    }

    pub fn toggle_selected_tunnel(&mut self) {
        let idx = self.selected_index;
        if self.tunnel_processes[idx].is_some() {
            self.stop_tunnel(idx);
        } else {
            self.start_tunnel(idx);
        }
    }

    pub fn check_port_health(&mut self) {
        for (idx, port_mapping) in self.ports.iter().enumerate() {
            let local_port = port_mapping.local_port();
            let addr = format!("127.0.0.1:{}", local_port);

            match TcpStream::connect_timeout(&addr.parse().unwrap(), Duration::from_millis(100)) {
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

    pub fn should_check_health(&self) -> bool {
        match self.last_health_check {
            None => true,
            Some(last) => last.elapsed() > Duration::from_secs(2),
        }
    }

    fn build_tunnel_command(&self, idx: usize) -> Command {
        let port_mapping = &self.ports[idx];
        let local_port = port_mapping.local_port();
        let remote_port = port_mapping.remote_port();

        match &self.tunnel_type {
            TunnelType::Ssh { host } => {
                let mut cmd = Command::new("ssh");
                cmd.args([
                    "-N",
                    "-L",
                    &format!("{}:localhost:{}", local_port, remote_port),
                    host,
                ]);
                cmd.stderr(Stdio::piped());
                cmd
            }
            TunnelType::Kubernetes {
                resource,
                namespace,
                context,
            } => {
                let mut cmd = Command::new("kubectl");
                cmd.arg("port-forward");

                if let Some(ns) = namespace {
                    cmd.args(["-n", ns]);
                }
                if let Some(ctx) = context {
                    cmd.args(["--context", ctx]);
                }

                cmd.arg(resource);
                cmd.arg(format!("{}:{}", local_port, remote_port));
                cmd.stderr(Stdio::piped());
                cmd
            }
        }
    }

    pub fn start_tunnel(&mut self, idx: usize) {
        let port_mapping = &self.ports[idx];
        let local_port = port_mapping.local_port();

        let mut cmd = self.build_tunnel_command(idx);

        match cmd.spawn() {
            Ok(mut child) => {
                std::thread::sleep(Duration::from_millis(200));

                match child.try_wait() {
                    Ok(Some(status)) => {
                        let stderr = child.stderr.take();
                        if let Some(stderr) = stderr {
                            let reader = BufReader::new(stderr);
                            let errors: Vec<String> =
                                reader.lines().filter_map(|line| line.ok()).collect();
                            if !errors.is_empty() {
                                self.status =
                                    format!("✗ Port {} failed: {}", local_port, errors.join("; "));
                            } else {
                                self.status = format!(
                                    "✗ Port {} exited with code {:?}",
                                    local_port,
                                    status.code()
                                );
                            }
                        } else {
                            self.status = format!(
                                "✗ Port {} failed with code {:?}",
                                local_port,
                                status.code()
                            );
                        }
                    }
                    Ok(None) => {
                        self.tunnel_processes[idx] = Some(child);
                        self.status = format!("✓ Port {} tunnel active", local_port);
                    }
                    Err(e) => {
                        self.status = format!("✗ Port {} error: {}", local_port, e);
                    }
                }
            }
            Err(e) => {
                self.status = format!("✗ Failed to start port {}: {}", local_port, e);
            }
        }
    }

    pub fn stop_tunnel(&mut self, idx: usize) {
        if let Some(mut child) = self.tunnel_processes[idx].take() {
            let _ = child.kill();
            let _ = child.wait();
            let local_port = self.ports[idx].local_port();
            self.status = format!("Stopped port {}", local_port);
        }
    }

    pub fn stop_all_tunnels(&mut self) {
        for process in &mut self.tunnel_processes {
            if let Some(mut child) = process.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}
