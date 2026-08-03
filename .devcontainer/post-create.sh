#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v gh &> /dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y gh
fi

# Claude Code CLI plus the caveman plugin. Kept non-fatal: a network hiccup here
# must not abort the whole container setup.
export PATH="$HOME/.local/bin:$PATH"

if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash || echo "WARN: Claude Code install failed, skipping."
fi

if command -v claude &> /dev/null; then
    if ! claude plugin marketplace list 2>/dev/null | grep -q "caveman"; then
        claude plugin marketplace add JuliusBrussee/caveman || echo "WARN: caveman marketplace add failed."
    fi
    if ! claude plugin list 2>/dev/null | grep -q "caveman@caveman"; then
        claude plugin install caveman@caveman --scope user || echo "WARN: caveman plugin install failed."
    fi
fi

python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,service]"
