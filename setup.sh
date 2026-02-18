#!/usr/bin/env bash
# Orze — one-command install
# Usage: curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
set -e

REPO="https://raw.githubusercontent.com/warlockee/orze/main"
DIR="orze"

mkdir -p "$DIR"

echo "Downloading orze..."
for f in farm.py bug_fixer.py AGENT.md RULES.md orze.yaml.example; do
  curl -sL "$REPO/$f" -o "$DIR/$f"
done

echo ""
echo "Done! orze installed to ./$DIR/"
echo ""
echo "Next: open any of your LLM cli and say:"
echo ""
echo "  do @orze/AGENT.md"
echo ""
