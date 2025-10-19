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
    pub ports: Vec<u16>,
    pub port_status: Vec<PortStatus>,
    pub ssh_processes: Vec<Option<Child>>,
    pub host: String,
    pub status: String,
    pub selected_index: usize,
    last_health_check: Option<Instant>,
}

impl App {
    pub fn new(host: String, ports: Vec<u16>) -> App {
        let port_count = ports.len();
        let ssh_processes = (0..port_count).map(|_| None).collect();
        App {
            ports,
            port_status: vec![PortStatus::Unknown; port_count],
            ssh_processes,
            host,
            status: String::from("j/k navigate • Enter toggle • q quit"),
            selected_index: 0,
            last_health_check: None,
        }
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
        if self.ssh_processes[idx].is_some() {
            self.stop_tunnel(idx);
        } else {
            self.start_tunnel(idx);
        }
    }

    pub fn check_port_health(&mut self) {

        for (idx, port) in self.ports.iter().enumerate() {
            let addr = format!("127.0.0.1:{}", port);

            // Check if SSH tunnel port is listening locally
            // Note: This verifies the tunnel is active, not if the remote service is responding
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

    pub fn should_check_health(&self) -> bool {
        match self.last_health_check {
            None => true,
            Some(last) => last.elapsed() > Duration::from_secs(2),
        }
    }

    pub fn start_tunnel(&mut self, idx: usize) {
        let port = self.ports[idx];
        let args = vec![
            "-N".to_string(),
            "-L".to_string(),
            format!("{}:localhost:{}", port, port),
            self.host.clone(),
        ];

        match Command::new("ssh")
            .args(&args)
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                std::thread::sleep(Duration::from_millis(200));

                match child.try_wait() {
                    Ok(Some(status)) => {
                        let stderr = child.stderr.take();
                        if let Some(stderr) = stderr {
                            let reader = BufReader::new(stderr);
                            let errors: Vec<String> = reader.lines()
                                .filter_map(|line| line.ok())
                                .collect();
                            if !errors.is_empty() {
                                self.status = format!("✗ Port {} failed: {}", port, errors.join("; "));
                            } else {
                                self.status = format!("✗ Port {} exited with code {:?}", port, status.code());
                            }
                        } else {
                            self.status = format!("✗ Port {} failed with code {:?}", port, status.code());
                        }
                    }
                    Ok(None) => {
                        self.ssh_processes[idx] = Some(child);
                        self.status = format!("✓ Port {} tunnel active", port);
                    }
                    Err(e) => {
                        self.status = format!("✗ Port {} error: {}", port, e);
                    }
                }
            }
            Err(e) => {
                self.status = format!("✗ Failed to start port {}: {}", port, e);
            }
        }
    }

    pub fn stop_tunnel(&mut self, idx: usize) {
        if let Some(mut child) = self.ssh_processes[idx].take() {
            let _ = child.kill();
            let _ = child.wait();
            let port = self.ports[idx];
            self.status = format!("Stopped port {}", port);
        }
    }

    pub fn stop_all_tunnels(&mut self) {
        for process in &mut self.ssh_processes {
            if let Some(mut child) = process.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}
