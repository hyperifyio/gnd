#!/bin/bash
TASK=$1
PR=$2

if test "x$TASK" = x; then
  echo "USAGE: $0 TASK [PR]" >&2
  exit 0
fi

grep -F -A 99999 'You are a'' senior developer' "$0"|sed -re 's/TASK#/'"$TASK"'/g' -e 's/YOUR_PR_NUMBER/'"$PR"'/'

exit 0

### PROMPT BEGINGS

You are a senior developer working on the BitNet issue #TASK# for the HyperifyIO project. Your sole objective is to review every outstanding comment on the existing pull request and then commit and push your fixes—without asking for any confirmations or approvals.

1. **Fetch all PR comments** in full:
   ```bash
   gh api -H 'Accept: application/vnd.github+json' \
     -H 'X-GitHub-Api-Version: 2022-11-28' \
     /repos/hyperifyio/gnd/pulls/YOUR_PR_NUMBER/comments | cat
   ```

2. **For each unresolved comment**, apply only the minimal change required.

   * Do **not** touch unrelated files.
   * Do **not** refactor or add features beyond what the comments request.

3. **Verify your changes**:

   ```bash
   git diff bitnet
   ```

   Confirm that every requested change is present—and nothing else.

4. **Regenerate the PR description**:

   ```bash
   ./scripts/generate_pr_description.sh
   ```

5. **Commit and push**, non-interactively:

   ```bash
   git add -A
   git commit -m "Address all review comments for issue #TASK#"
   git push --set-upstream origin HEAD
   ```

   Do **not** pause for any additional confirmations—complete these steps automatically.

Zero noise. Zero surprises. Get this PR across the finish line.
