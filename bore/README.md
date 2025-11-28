# Bore - SSH & Kubernetes Port Forwarding TUI

Ultra-lightweight terminal UI for managing SSH and Kubernetes port forwards with real-time health checks.

## Features

- 🚀 **Presets** - Save your common configurations
- ☸ **Kubernetes Support** - Port-forward to pods, services, and deployments
- ⚡ **SSH Tunnels** - Classic SSH port forwarding with your SSH config
- 💚 **Tunnel Monitoring** - Real-time per-port tunnel status
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

You'll be prompted to choose tunnel type (SSH or Kubernetes), then target and ports.

### Using Presets (Recommended)

1. **Create config file:**

```bash
cargo run -- --init-config
```

2. **Edit** `~/.config/bore/bore.toml`:

```toml
# SSH tunnel preset
[presets.work]
type = "ssh"
host = "dev"
ports = [3001, 8080, 4002]

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

# Port mapping (local:remote)
[presets.api]
type = "k8s"
resource = "deploy/api-server"
namespace = "backend"
ports = [
    8080,
    { local = 9090, remote = 8081 }
]
```

3. **Run with preset:**

```bash
cargo run -- --preset redis
```

## CLI Commands

```bash
bore --preset <name>      # Use a saved preset
bore --list-presets       # Show all available presets
bore --init-config        # Create example config file
bore                      # Interactive mode
```

## TUI Controls

- `j/k` (or `↑/↓`) - **Navigate** between ports
- `Enter/Space` - **Toggle** selected tunnel (start/stop)
- `q` - **Quit** (stops all tunnels)

## Tunnel Status

Each port shows real-time tunnel status:

- `▸` **Green** - Tunnel port accessible
- `▸` **Red** - Tunnel port not responding
- `◐` **Yellow** - Checking status
- `▹` **Gray** - Tunnel inactive

## Kubernetes Resources

Supported resource types for `resource` field:

- `pod/my-pod` - Forward to a specific pod
- `svc/my-service` - Forward to a service
- `deploy/my-deploy` - Forward to a deployment

Optional fields:

- `namespace` - Kubernetes namespace (default: current namespace)
- `context` - Kubernetes context (default: current context)

## Notes

- SSH tunnels work with your `~/.ssh/config`
- K8s port-forwards require `kubectl` in your PATH
- Status shows if the tunnel is listening locally, not if the remote service is healthy
