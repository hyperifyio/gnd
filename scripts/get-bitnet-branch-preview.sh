#!/bin/bash
TASK=$1
if test "x$TASK" = x; then
  TASK=$(./scripts/get-current-task-number.sh)
fi

if [ -z "$TASK" ]; then
  echo "USAGE: $0 TASK" >&2
  exit 1
fi

# Check current PR number
PR=$(./scripts/get-current-pr-number.sh)

echo '**You are a senior developer working on the BitNet issue #TASK# and PR #PR# for the HyperifyIO project.**'

# Check current task info
echo
echo '### Current Task & Scope ###'
echo
./scripts/get-current-task.sh
echo
echo  ----------------------------
echo

echo '### Current Feature & Goal ###'
echo
./scripts/get-bitnet-task.sh
echo
echo  ------------------------------
echo

grep -F -A 99999 'Your'' sole objective' "$0" \
  | sed -e 's/#TASK#/'"$TASK"'/g' \
  | sed -e 's/#PR#/'"$PR"'/g'

exit 0

### PROMPT BEGINS
Your sole objective is to:

1. **Preview all changes** in the issue branch relative to `bitnet`: `git diff bitnet`, and `git diff --cached` and `git diff`
   - You should also preview only the implementation changes: `./scripts/bitnet-get-current-implementation-changes.sh`
2. **Review the goal** of issue #TASK# (use `./scripts/get-current-task.sh|cat` and/or `gh` to view info).
3. **Verify** that every change shown by `git diff bitnet` is fully aligned with the stated goal of issue #TASK#.
4. **Ensure** no unrelated files or off-task modifications are included.
5. **Confirm** there are **no duplicate implementations**—verify that functionality isn’t already present elsewhere in the codebase before proceeding.

After verifying, report back with either a clean confirmation or a list of any discrepancies or duplicates found.
