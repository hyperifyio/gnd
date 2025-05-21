#!/bin/bash

# Function to safely extract benchmark values
extract_benchmark() {
    local pattern=$1
    local value=$(grep "$pattern" benchmark_results.txt | head -n 1 | awk '{print $'$2'}')
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

# Function to extract timing values
extract_timing() {
    local pattern=$1
    local value=$(grep "$pattern" benchmark_results.txt | head -n 1 | awk '{print $3}')
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

# Function to get previous coverage from git history
get_previous_coverage() {
    local previous_coverage=$(git log --all | grep -FA 1 "Current coverage:" | grep -Eo 'Current coverage:.*'|head -n 1|tr -d ' '|awk -F: '{print $2}')
    if [ -z "$previous_coverage" ]; then
        echo "N/A"
    else
        echo "$previous_coverage"
    fi
}

# Generate test coverage report
echo "Generating test coverage report..."
go test ./pkg/bitnet/... -coverprofile=coverage.out
COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')
PREVIOUS_COVERAGE=$(get_previous_coverage)

# Run benchmarks
echo "Running benchmarks..."
./scripts/run_benchmarks.sh > benchmark_results.txt

# Check if benchmark results file exists and has content
if [ ! -s benchmark_results.txt ]; then
    echo "Warning: No benchmark results found. Using placeholder values."
    # Set default values for missing benchmarks
    NEW_TENSOR_ALLOCS="N/A"
    GET_SET_ALLOCS="N/A"
    PARALLEL_ALLOCS="N/A"
    BASIC_OPS_TIME="N/A"
    PARALLEL_OPS_TIME="N/A"
    LARGE_OPS_TIME="N/A"
    MODEL_LOAD_TIME="N/A"
    MODEL_LOAD_ALLOCS="N/A"
    MODEL_INFER_TIME="N/A"
    MODEL_INFER_ALLOCS="N/A"
    TERNARY_WEIGHTS_TIME="N/A"
    TERNARY_WEIGHTS_ALLOCS="N/A"
else
    # Extract tensor benchmark results
    NEW_TENSOR_ALLOCS=$(extract_benchmark "BenchmarkNewTensor/shape_\[100\]" 5)
    GET_SET_ALLOCS=$(extract_benchmark "BenchmarkTensor_Get/2D_access" 5)
    PARALLEL_ALLOCS=$(extract_benchmark "BenchmarkTensor_ParallelForEach/100x100" 5)

    # Extract timing values
    BASIC_OPS_TIME=$(extract_timing "BenchmarkTensor_Get/2D_access")
    PARALLEL_OPS_TIME=$(extract_timing "BenchmarkTensor_ParallelForEach/100x100")
    LARGE_OPS_TIME=$(extract_timing "BenchmarkNewTensor/shape_\[100_100\]")

    # Extract BitNet model benchmark results
    MODEL_LOAD_TIME=$(extract_timing "BenchmarkModel_LoadWeights")
    MODEL_LOAD_ALLOCS=$(extract_benchmark "BenchmarkModel_LoadWeights" 5)
    MODEL_INFER_TIME=$(extract_timing "BenchmarkModel_Infer")
    MODEL_INFER_ALLOCS=$(extract_benchmark "BenchmarkModel_Infer" 5)
    TERNARY_WEIGHTS_TIME=$(extract_timing "BenchmarkModel_ReadTernaryWeights")
    TERNARY_WEIGHTS_ALLOCS=$(extract_benchmark "BenchmarkModel_ReadTernaryWeights" 5)

    # Extract BitLinear benchmark results
    BITLINEAR_TIME=$(extract_timing "BenchmarkBitLinear")
    BITLINEAR_ALLOCS=$(extract_benchmark "BenchmarkBitLinear" 5)

    # Set default values for unimplemented benchmarks
    if [ "$MODEL_INFER_TIME" = "N/A" ]; then
        MODEL_INFER_TIME="N/A (TODO #190)"
    fi
    if [ "$MODEL_INFER_ALLOCS" = "N/A" ]; then
        MODEL_INFER_ALLOCS="N/A (TODO #190)"
    fi
fi

# Generate PR description
cat << EOF > pr_description.md
## Changes
- [ ] List of specific changes made
- [ ] Include file paths and line numbers for major changes
- [ ] Reference related issues/tickets

## Test Coverage
- Current coverage: ${COVERAGE}
- Coverage changes: ${PREVIOUS_COVERAGE} → ${COVERAGE}

## Performance Metrics
### Memory Usage
#### Tensor Operations
- Allocations per operation:
  - New tensor creation: ${NEW_TENSOR_ALLOCS} allocs/op
  - Get/Set operations: ${GET_SET_ALLOCS} allocs/op
  - Parallel operations: ${PARALLEL_ALLOCS} allocs/op
  - BitLinear operations: ${BITLINEAR_ALLOCS} allocs/op

#### BitNet Model Operations
- Allocations per operation:
  - Model weights loading: ${MODEL_LOAD_ALLOCS} allocs/op
  - Model inference: ${MODEL_INFER_ALLOCS} allocs/op (TODO #190)
  - Ternary weights reading: ${TERNARY_WEIGHTS_ALLOCS} allocs/op

### CPU Performance
#### Tensor Operations
- Operation timing:
  - Basic operations: ${BASIC_OPS_TIME} ns/op
  - Parallel operations: ${PARALLEL_OPS_TIME} ns/op
  - Large tensor operations: ${LARGE_OPS_TIME} ns/op
  - BitLinear operations: ${BITLINEAR_TIME} ns/op

#### BitNet Model Operations
- Operation timing:
  - Model weights loading: ${MODEL_LOAD_TIME} ns/op
  - Model inference: ${MODEL_INFER_TIME} ns/op (TODO #190)
  - Ternary weights reading: ${TERNARY_WEIGHTS_TIME} ns/op

## Areas for Improvement
### High Priority
- [ ] Optimize memory allocations in model operations (TODO #191)
- [ ] Implement proper self-attention (TODO #186)

### Medium Priority
- [ ] Improve error handling in model operations (TODO #192)
- [ ] Add more comprehensive benchmarks (TODO #192)
- [ ] Enhance documentation
- [ ] Implement proper feed-forward network (TODO #187)

### Low Priority
- [ ] Consider SIMD optimizations (TODO #191)
- [ ] Add more model operations (TODO #190)
- [ ] Improve test organization (TODO #192)
- [ ] Implement proper output generation (TODO #189)

Closes #201
EOF

echo "PR description generated in pr_description.md" 
