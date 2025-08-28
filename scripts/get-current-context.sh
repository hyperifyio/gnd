#!/usr/bin/env bash
set -euo pipefail

# Usage: ./build-task-prompt.sh [TASK_NUMBER]
#
# If no TASK_NUMBER is provided, fetch the current task.
TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  TASK="$(./scripts/get-current-task-number.sh)"
fi

if [[ -z "$TASK" ]]; then
  echo "USAGE: $0 TASK_NUMBER" >&2
  exit 1
fi

# Fetch current PR number (optional, for richer context)
PR="$(./scripts/get-current-pr-number.sh || echo "N/A")"

# Header: who & where
cat <<EOF
**You are a senior developer working on:**
- BitNet Issue  **#${TASK}**
- Pull Request **#${PR}** (branch: $(git rev-parse --abbrev-ref HEAD))

---

## 1) Task Details & Scope
$(./scripts/get-current-task.sh)

---

## 2) Feature & Goal Overview
$(./scripts/get-bitnet-task.sh)

---

## 3) Implementation Diff
- Unstaged changes:  
  \`\`\`bash
  git diff
  \`\`\`
- Staged changes:  
  \`\`\`bash
  git diff --cached
  \`\`\`
- All changes against \`bitnet\` branch:  
  \`\`\`bash
  git diff origin/bitnet...HEAD
  \`\`\`
- Only implementation lines (ignoring tests):
  \`\`\`bash
  ./scripts/bitnet-get-current-implementation-changes.sh
  \`\`\`

---

EOF

