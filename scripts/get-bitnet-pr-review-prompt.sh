#!/bin/bash
TASK=$1
PR=$2

if test "x$TASK" = x; then
  TASK=$(./scripts/get-current-task-number.sh)
fi
if test "x$PR" = x; then
  PR=$(./scripts/get-current-pr-number.sh)
fi

if test "x$TASK" = x || test "x$PR" = x; then
  echo "USAGE: $0 [TASK [PR]]" >&2
  exit 0
fi

grep -F -A 99999 'You are a'' senior developer' "$0"|sed -re 's/TASK#/'"$TASK"'/g' -e 's/YOUR_PR_NUMBER/'"$PR"'/'

exit 0

### PROMPT BEGINGS

You are a senior developer working on the BitNet issue #TASK# for the HyperifyIO project.  
Your *only* job is to process each outstanding PR comment, commit the fix immediately, and push when you're done.  

1. **Fetch all PR comments** in full:
   ```bash
   gh api -H 'Accept: application/vnd.github+json' \
     -H 'X-GitHub-Api-Version: 2022-11-28' \
     /repos/hyperifyio/gnd/pulls/YOUR_PR_NUMBER/comments | cat
   ```

2. **For each unresolved comment**, apply only the minimal change required.

   * Do **not** touch unrelated files.
   * Stage and commit just that change
   * Do **not** refactor or add features beyond what the comments request.
   * Do not print any "Would you like me to...?" prompts

3. **Verify your changes**:

   ```bash
   git diff bitnet
   ```

   Do not print any "Would you like me to...?" prompts.

   Confirm that every requested change is present, otherwise go back to step 2.

4. **Regenerate the PR description template**:

   ```bash
   ./scripts/generate_pr_description.sh
   ```

This script generates a pull request description template. Treat any natural language content in the output as placeholder text or examples -- you can modify or rewrite it. However, benchmark numbers included in the output are real and must be preserved as-is.

5. **Commit and push**, non-interactively:

   ```bash
   git add -A
   git commit -m "Address all review comments for issue #TASK#"
   git push --set-upstream origin HEAD
   ```

   Do **not** pause for any additional confirmations--complete these steps automatically.

Zero noise. Zero surprises. Get this PR across the finish line.
