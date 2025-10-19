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
    pub host: String,
    pub ssh_process: Option<Child>,
    pub status: String,
    last_health_check: Option<Instant>,
}

impl App {
    pub fn new(host: String, ports: Vec<u16>) -> App {
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

    pub fn check_port_health(&mut self) {
        if self.ssh_process.is_none() {
            for status in &mut self.port_status {
                *status = PortStatus::Unknown;
            }
            return;
        }

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
        if self.ssh_process.is_none() {
            return false;
        }

        match self.last_health_check {
            None => true,
            Some(last) => last.elapsed() > Duration::from_secs(2),
        }
    }

    pub fn start_ssh(&mut self) {
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
                                self.status = format!("✗ SSH failed: {}", errors.join("; "));
                            } else {
                                self.status = format!("✗ SSH exited with code {:?}", status.code());
                            }
                        } else {
                            self.status = format!("✗ SSH failed with code {:?}", status.code());
                        }
                    }
                    Ok(None) => {
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

    pub fn stop_ssh(&mut self) {
        if let Some(mut child) = self.ssh_process.take() {
            let _ = child.kill();
            self.status = String::from("Press 's' to start, 'q' to quit");
        }
    }
}
