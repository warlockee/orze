#!/usr/bin/env bash
# Orze — zero-dependency setup
# Requirements: bash + curl (present on all Linux/macOS systems)
# 100% non-interactive.
#
# Usage:
#   curl -sL https://orze.ai/setup.sh | bash
#   curl -sL https://orze.ai/setup.sh | bash -s /path/to/project
set -euo pipefail

# --- Colors ---------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${GREEN}[+]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[!]${RESET} $*"; }
step()  { echo -e "${BOLD}${CYAN}==> $*${RESET}"; }

# --- Install uv if missing ------------------------------------------------
step "Checking for uv..."
if command -v uv &>/dev/null; then
    info "uv already installed ($(uv --version))"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the env so uv is on PATH for the rest of this script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        warn "uv installed but not on PATH. Add ~/.local/bin to your PATH."
        exit 1
    fi
    info "uv installed ($(uv --version))"
fi

# --- Install orze via uv tool ---------------------------------------------
step "Installing orze..."
uv tool install --quiet --force orze
info "orze installed ($(orze --version 2>/dev/null || echo 'unknown'))"

# --- Auto-detect shared storage -------------------------------------------
PROJECT_PATH="${1:-}"

if [ -z "$PROJECT_PATH" ]; then
    step "Detecting shared storage..."
    # Linux: check /proc/mounts for network filesystems
    if [ -f /proc/mounts ]; then
        SHARED_MOUNT=$(awk '$3 ~ /^(lustre|nfs|nfs4|efs|gpfs|beegfs|glusterfs|pvfs2|fuse\.s3fs|fuse\.goofys)$/ || $1 ~ /@tcp:/ { print $2 }' /proc/mounts | head -1)
    fi
    # macOS: check mount for nfs
    if [ -z "${SHARED_MOUNT:-}" ] && command -v mount &>/dev/null; then
        SHARED_MOUNT=$(mount | awk '$5 == "nfs" || $5 == "nfs4" { print $3 }' | head -1)
    fi
    if [ -n "${SHARED_MOUNT:-}" ]; then
        info "Detected shared storage: ${SHARED_MOUNT}"
        PROJECT_PATH="$SHARED_MOUNT"
    else
        info "No shared storage detected, using current directory"
        PROJECT_PATH="."
    fi
fi

# --- Initialize project ---------------------------------------------------
step "Initializing project at ${PROJECT_PATH}..."
if ! orze --init "$PROJECT_PATH"; then
    warn "orze --init failed. Check the output above for details."
    exit 1
fi

# --- Done ------------------------------------------------------------------
echo ""
info "Setup complete!"
echo -e "  ${BOLD}Next:${RESET} cd ${PROJECT_PATH} && orze --check"
