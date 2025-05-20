#!/bin/bash
TASK=$1

if [ -z "$TASK" ]; then
  echo "USAGE: $0 TASK" >&2
  exit 1
fi

grep -F -A 99999 'You'' are a ' "$0" \
  | sed -e 's/#TASK#/'"$TASK"'/g'

exit 0

### PROMPT BEGINS
You are a senior developer working on the BitNet issue #TASK# for the HyperifyIO project. Your sole objective is to:

1. **Preview all changes** in the issue branch relative to `bitnet`: `git diff bitnet`
2. **Review the goal** of issue #TASK# (use `gh` to view the issue).
3. **Verify** that every change shown by `git diff bitnet` is fully aligned with the stated goal of issue #TASK#.
4. **Ensure** no unrelated files or off-task modifications are included.
5. **Confirm** there are **no duplicate implementations**—verify that functionality isn’t already present elsewhere in the codebase before proceeding.

After verifying, report back with either a clean confirmation or a list of any discrepancies or duplicates found.
