# Bore - Simple SSH Port Forwarding TUI

Ultra-lightweight terminal UI for managing SSH port forwards with real-time health checks.

## Features

- 🚀 **Presets** - Save your common configurations
- 💚 **Tunnel Monitoring** - Real-time per-port tunnel status
- ⚡ **Fast** - Lightweight TUI with minimal dependencies
- 🎨 **Futuristic UI** - Clean, color-coded status indicators

## Setup

```bash
cd bore
cargo build --release
```

## Quick Start

### Interactive Mode

```bash
cargo run
```

You'll be prompted for host and ports.

### Using Presets (Recommended)

1. **Create config file:**
```bash
cargo run -- --init-config
```

2. **Edit** `~/.config/bore/config.toml`:
```toml
[presets.work]
host = "dev"
ports = [3001, 8080, 4002]

[presets.staging]
host = "user@staging-server"
ports = [5432, 8000]
```

3. **Run with preset:**
```bash
cargo run -- --preset work
```

## CLI Commands

```bash
bore --preset <name>      # Use a saved preset
bore --list-presets       # Show all available presets
bore --init-config        # Create example config file
bore                      # Interactive mode (prompts for host/ports)
```

## TUI Controls

- `s` - **Start** SSH tunnel
- `x` - **Stop** tunnel
- `q` - **Quit**

## Tunnel Status

Each port shows real-time tunnel status:
- `▸` **Green** - Tunnel port accessible
- `▸` **Red** - Tunnel port not responding
- `◐` **Yellow** - Checking status
- `▹` **Gray** - Tunnel inactive

**Note:** Status shows if the SSH tunnel is listening, not if the remote service is up. This is a limitation of how SSH port forwarding works - the tunnel accepts connections locally even if the remote service is down.

Works with your SSH config!
