#!/bin/bash
# Finish one curriculum item: format, lint, test, tick its box, commit.
# usage: scripts/done.sh <section/topic> "<commit message>"
set -e
topic="$1"
touch "$(dirname "$topic")/__init__.py" "$topic/__init__.py"
ruff format -q "$topic"
ruff check --fix -q "$topic"
pytest -q "$topic" 2>&1 | tail -3
sed -i '' "s#^- \[ \] \`$topic\`#- [x] \`$topic\`#" CURRICULUM.md
git add -A
git commit -qm "$2"
echo "committed: $2"
