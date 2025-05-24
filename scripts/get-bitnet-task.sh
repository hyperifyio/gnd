#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get BitNet task details
echo -e "${YELLOW}Fetching BitNet task details...${NC}"
BITNET_TASK=$(gh issue view 170 --json title,body,state,labels 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Could not fetch BitNet task #170${NC}"
    echo "Make sure you're authenticated with GitHub CLI and the issue exists"
    exit 1
fi

# Extract and display BitNet task information
TITLE=$(echo "$BITNET_TASK" | jq -r '.title')
STATE=$(echo "$BITNET_TASK" | jq -r '.state')
LABELS=$(echo "$BITNET_TASK" | jq -r '.labels[].name' | tr '\n' ', ' | sed 's/, $//')

echo -e "\n${GREEN}BitNet Task:${NC}"
echo -e "Issue #170: $TITLE"
echo -e "State: $STATE"
echo -e "Labels: $LABELS"
echo -e "\n${YELLOW}Description:${NC}"
echo "$BITNET_TASK" | jq -r '.body'

# List open tasks first
echo -e "\n${BLUE}Open BitNet Tasks:${NC}"
gh issue list --label "bitnet,task" --state open --json number,title,state --jq '.[] | "\(.number): \(.title) (\(.state))"' | while read -r line; do
    if [[ $line =~ ^170: ]]; then
        echo -e "${GREEN}$line${NC}"
    else
        echo "$line"
    fi
done

# Then list closed tasks
echo -e "\n${BLUE}Closed BitNet Tasks:${NC}"
gh issue list --label "bitnet,task" --state closed --json number,title,state --jq '.[] | "\(.number): \(.title) (\(.state))"' | while read -r line; do
    echo -e "${RED}$line${NC}"
done 