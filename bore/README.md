# Bore - Simple SSH Port Forwarding TUI

Ultra-lightweight terminal UI for managing SSH port forwards with real-time health checks.

## Features

- 🚀 **Presets** - Save your common configurations
- 💚 **Health Checks** - Real-time per-port status monitoring
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

## Health Status

Each port shows real-time status:
- `●` **Green** - Service healthy and responding
- `●` **Red** - Tunnel up, but service down
- `◐` **Yellow** - Checking status
- `▹` **Gray** - Tunnel inactive

Works with your SSH config!
