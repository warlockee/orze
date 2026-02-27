#!/usr/bin/env bash
# Orze — one-command install
# Usage: curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
set -e

echo "Installing orze..."
pip install orze

echo ""
echo "Done! orze is installed."
echo ""
echo "Initialize your project:"
echo "  orze --init"
echo ""
echo "Then launch:"
echo "  orze"
echo ""
