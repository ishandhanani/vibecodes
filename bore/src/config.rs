use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub presets: HashMap<String, Preset>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Preset {
    pub host: String,
    pub ports: Vec<u16>,
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

        toml::from_str(&contents)
            .map_err(|e| format!("Failed to parse config file: {}", e))
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

        let example = r#"# Bore SSH Port Forward Presets
# Location: ~/.config/bore/bore.toml

[presets.work]
host = "dev"
ports = [3001, 8080, 4002]

[presets.staging]
host = "user@staging-server"
ports = [5432, 8000]

# Add your own presets here
"#;

        fs::write(&config_path, example)
            .map_err(|e| format!("Failed to write example config: {}", e))?;

        Ok(())
    }
}

fn get_config_path() -> Result<PathBuf, String> {
    let home_dir = dirs::home_dir()
        .ok_or_else(|| "Could not find home directory".to_string())?;

    Ok(home_dir.join(".config").join("bore").join("bore.toml"))
}

pub fn get_config_path_display() -> String {
    get_config_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "~/.config/bore/bore.toml".to_string())
}
