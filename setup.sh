#!/usr/bin/env bash
# Orze — one-command install
# Usage: curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
set -e

REPO="https://raw.githubusercontent.com/warlockee/orze/main"
DIR="orze"

mkdir -p "$DIR"

echo "Downloading orze..."
for f in farm.py AGENT.md RULES.md orze.yaml.example; do
  curl -sL "$REPO/$f" -o "$DIR/$f"
done

echo ""
echo "Done! orze installed to ./$DIR/"
echo ""
echo "Next: open Claude Code and say:"
echo ""
echo "  @orze/AGENT.md set up and run experiments for this project"
echo ""
