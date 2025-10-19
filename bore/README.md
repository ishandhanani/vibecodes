# Bore - Simple SSH Port Forwarding TUI

Ultra-lightweight terminal UI for managing SSH port forwards.

## Setup

```bash
cd bore
cargo build --release
```

## Run

```bash
cargo run
```

You'll be prompted for:
1. SSH host (e.g., `dev` or `user@dev`)
2. Ports to forward (e.g., `3001,8080,4002`)

## Usage

Once in the TUI:
- Press `s` to **start** SSH tunnel
- Press `x` to **stop** tunnel
- Press `q` to **quit**

## Example

```bash
$ cargo run
SSH host (e.g., 'dev' or 'user@dev'): username@dev
Ports to forward (comma-separated, e.g., '3001,8080,4002'): 3001,8080,4002
```

The TUI will then manage your SSH port forwards. Works with your SSH config!
