#!/bin/bash

# Get current branch name
BRANCH_NAME=$(git branch --show-current)

# Extract issue number from branch name (format: number-description)
ISSUE_NUMBER=$(echo "$BRANCH_NAME" | grep -o '^[0-9]\+')

if [ -z "$ISSUE_NUMBER" ]; then
    echo "Error: Could not detect issue number from branch name" >&2
    echo "Expected branch name format: <number>-description" >&2
    exit 1
fi

# Just print the number
echo "$ISSUE_NUMBER" 