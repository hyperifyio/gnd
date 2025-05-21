#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current branch name
BRANCH_NAME=$(git branch --show-current)

# Extract issue number from branch name (format: number-description)
ISSUE_NUMBER=$(echo "$BRANCH_NAME" | grep -o '^[0-9]\+')

if [ -z "$ISSUE_NUMBER" ]; then
    echo -e "${RED}Error: Could not detect issue number from branch name${NC}"
    echo "Expected branch name format: <number>-description"
    exit 1
fi

# Get issue details using GitHub CLI
echo -e "${YELLOW}Fetching details for issue #$ISSUE_NUMBER...${NC}"
ISSUE_DETAILS=$(gh issue view "$ISSUE_NUMBER" --json title,body,state,labels 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Could not fetch issue #$ISSUE_NUMBER${NC}"
    echo "Make sure you're authenticated with GitHub CLI and the issue exists"
    exit 1
fi

# Extract and display issue information
TITLE=$(echo "$ISSUE_DETAILS" | jq -r '.title')
STATE=$(echo "$ISSUE_DETAILS" | jq -r '.state')
LABELS=$(echo "$ISSUE_DETAILS" | jq -r '.labels[].name' | tr '\n' ', ' | sed 's/, $//')

echo -e "\n${GREEN}Current Task:${NC}"
echo -e "Issue #$ISSUE_NUMBER: $TITLE"
echo -e "State: $STATE"
echo -e "Labels: $LABELS"
echo -e "\n${YELLOW}Description:${NC}"
echo "$ISSUE_DETAILS" | jq -r '.body' 