#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BENCH_DIRS=("./pkg/bitnet/tensor" "./pkg/bitnet/model")
PROFILE_DIR="profiles"
THRESHOLDS_FILE=".cursor/rules/bitnet-performance.mdc"

# Create profile directory if it doesn't exist
mkdir -p "$PROFILE_DIR"

echo -e "${YELLOW}Running performance tests...${NC}"

# Run benchmarks for each directory
for BENCH_DIR in "${BENCH_DIRS[@]}"; do
    echo -e "\n${YELLOW}Running benchmarks in $BENCH_DIR...${NC}"
    
    # Run benchmarks with memory profiling
    echo -e "\n${YELLOW}Running memory benchmarks...${NC}"
    cd "$(dirname "$0")/.." && go test -timeout 30s -bench=. -benchmem -memprofile="$PROFILE_DIR/mem.prof" "$BENCH_DIR"

    # Run benchmarks with CPU profiling
    echo -e "\n${YELLOW}Running CPU benchmarks...${NC}"
    cd "$(dirname "$0")/.." && go test -timeout 30s -bench=. -cpuprofile="$PROFILE_DIR/cpu.prof" "$BENCH_DIR"

    # Run performance checks
    echo -e "\n${YELLOW}Running performance checks...${NC}"
    cd "$(dirname "$0")/.." && go test -timeout 30s -bench=. -benchmem "$BENCH_DIR" | while read -r line; do
        if [[ $line =~ ^Benchmark ]]; then
            echo -e "${GREEN}$line${NC}"
        elif [[ $line =~ allocs/op ]]; then
            allocs=$(echo "$line" | awk '{print $3}')
            if (( $(echo "$allocs > 10" | bc -l) )); then
                echo -e "${RED}High allocation rate: $allocs allocs/op${NC}"
            else
                echo -e "${GREEN}$line${NC}"
            fi
        elif [[ $line =~ B/op ]]; then
            bytes=$(echo "$line" | awk '{print $3}')
            if (( $(echo "$bytes > 1024" | bc -l) )); then
                echo -e "${RED}High memory usage: $bytes B/op${NC}"
            else
                echo -e "${GREEN}$line${NC}"
            fi
        elif [[ $line =~ ns/op ]]; then
            ns=$(echo "$line" | awk '{print $3}')
            if (( $(echo "$ns > 1000" | bc -l) )); then
                echo -e "${RED}Slow operation: $ns ns/op${NC}"
            else
                echo -e "${GREEN}$line${NC}"
            fi
        else
            echo "$line"
        fi
    done
done

echo -e "\n${GREEN}Performance testing complete!${NC}"

# Run memory benchmarks
echo -e "\033[1;33mRunning memory benchmarks...\033[0m"
go test -timeout 30s -bench=. -benchmem ./pkg/bitnet/tensor/...

# Run CPU benchmarks
echo -e "\033[1;33mRunning CPU benchmarks...\033[0m"
go test -timeout 30s -bench=. ./pkg/bitnet/tensor/...

# Run performance checks
echo -e "\033[1;33mRunning performance checks...\033[0m"
go test -timeout 30s -bench=. -benchmem ./pkg/bitnet/tensor/...

echo -e "\033[0;32mPerformance testing complete!\033[0m" 
