#!/usr/bin/env bash
# Point git at the repo's versioned hooks. Run once per clone:
#   bash scripts/install_hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
chmod +x scripts/git-hooks/*
git config core.hooksPath scripts/git-hooks
echo "hooks installed: core.hooksPath -> scripts/git-hooks"
