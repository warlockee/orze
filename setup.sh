#!/usr/bin/env bash
# Orze — one-command install
# Usage: curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
set -e

echo "Installing orze..."
pip install --quiet orze

echo ""
orze --init
