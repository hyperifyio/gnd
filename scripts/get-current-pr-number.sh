#!/bin/bash

# Get PR number for current branch using GitHub CLI
PR_NUMBER=$(gh pr view --json number --jq .number 2>/dev/null)

if [ $? -ne 0 ] || [ -z "$PR_NUMBER" ]; then
    echo "Error: Could not detect PR number for current branch" >&2
    echo "Make sure you're authenticated with GitHub CLI and the branch has an associated PR" >&2
    exit 1
fi

# Just print the number
echo "$PR_NUMBER" 