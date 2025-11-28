use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum TunnelType {
    Ssh {
        host: String,
    },
    #[serde(rename = "k8s")]
    Kubernetes {
        resource: String,
        #[serde(default)]
        namespace: Option<String>,
        #[serde(default)]
        context: Option<String>,
    },
}

impl TunnelType {
    pub fn display_name(&self) -> String {
        match self {
            TunnelType::Ssh { host } => host.clone(),
            TunnelType::Kubernetes {
                resource,
                namespace,
                ..
            } => match namespace {
                Some(ns) => format!("{}/{}", ns, resource),
                None => resource.clone(),
            },
        }
    }

    pub fn type_label(&self) -> &'static str {
        match self {
            TunnelType::Ssh { .. } => "SSH",
            TunnelType::Kubernetes { .. } => "K8S",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct Preset {
    #[serde(flatten)]
    pub tunnel_type: TunnelType,
    pub ports: Vec<PortMapping>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum PortMapping {
    Single(u16),
    Mapped { local: u16, remote: u16 },
}

impl PortMapping {
    pub fn local_port(&self) -> u16 {
        match self {
            PortMapping::Single(p) => *p,
            PortMapping::Mapped { local, .. } => *local,
        }
    }

    pub fn remote_port(&self) -> u16 {
        match self {
            PortMapping::Single(p) => *p,
            PortMapping::Mapped { remote, .. } => *remote,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub presets: HashMap<String, Preset>,
}

impl Config {
    pub fn load() -> Result<Config, String> {
        let config_path = get_config_path()?;

        if !config_path.exists() {
            return Ok(Config {
                presets: HashMap::new(),
            });
        }

        let contents = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;

        toml::from_str(&contents).map_err(|e| format!("Failed to parse config file: {}", e))
    }

    pub fn get_preset(&self, name: &str) -> Option<&Preset> {
        self.presets.get(name)
    }

    pub fn create_example_config() -> Result<(), String> {
        let config_path = get_config_path()?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }

        let example = r#"# Bore Port Forward Presets
# Location: ~/.config/bore/bore.toml

# SSH tunnel preset
[presets.work]
type = "ssh"
host = "dev"
ports = [3001, 8080, 4002]

[presets.staging]
type = "ssh"
host = "user@staging-server"
ports = [5432, 8000]

# Kubernetes port-forward presets
[presets.redis]
type = "k8s"
resource = "svc/redis"
namespace = "default"
ports = [6379]

[presets.postgres]
type = "k8s"
resource = "pod/postgres-0"
namespace = "database"
context = "prod-cluster"
ports = [5432]

# Port mapping example (local:remote)
[presets.api]
type = "k8s"
resource = "deploy/api-server"
namespace = "backend"
ports = [
    8080,
    { local = 9090, remote = 8081 }
]
"#;

        fs::write(&config_path, example)
            .map_err(|e| format!("Failed to write example config: {}", e))?;

        Ok(())
    }
}

fn get_config_path() -> Result<PathBuf, String> {
    let home_dir = dirs::home_dir().ok_or_else(|| "Could not find home directory".to_string())?;

    Ok(home_dir.join(".config").join("bore").join("bore.toml"))
}

pub fn get_config_path_display() -> String {
    get_config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "~/.config/bore/bore.toml".to_string())
}
