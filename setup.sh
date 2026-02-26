#!/usr/bin/env bash
# Orze — one-command install
# Usage: curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
set -e

REPO="https://raw.githubusercontent.com/warlockee/orze/main"
DIR="orze"

mkdir -p "$DIR"

echo "Downloading orze..."
FILES=(
  "farm.py" "bug_fixer.py" "bot.py" "idea_lake.py" "orze_gc.py" 
  "rebuild_lake.py" "manual_notify.py" "AGENT.md" "RULES.md" 
  "orze.yaml.example" "requirements.txt" "pyproject.toml"
)

for f in "${FILES[@]}"; do
  curl -sL "$REPO/$f" -o "$DIR/$f"
done

echo ""
echo "Done! orze installed to ./$DIR/"
echo ""
echo "Next: initialize your project with:"
echo ""
echo "  python3 orze/farm.py --init"
echo ""
