#!/bin/bash

# Generate test coverage report
echo "Generating test coverage report..."
go test ./pkg/bitnet/... -coverprofile=coverage.out
COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')

# Run benchmarks
echo "Running benchmarks..."
./scripts/run_benchmarks.sh > benchmark_results.txt

# Extract benchmark results
NEW_TENSOR_ALLOCS=$(grep "BenchmarkNewTensor/shape_\[100\]" benchmark_results.txt | head -n 1 | awk '{print $5}')
GET_SET_ALLOCS=$(grep "BenchmarkTensor_Get/2D_access" benchmark_results.txt | head -n 1 | awk '{print $5}')
PARALLEL_ALLOCS=$(grep "BenchmarkTensor_ParallelForEach/100x100" benchmark_results.txt | head -n 1 | awk '{print $5}')

BASIC_OPS_TIME=$(grep "BenchmarkTensor_Get/2D_access" benchmark_results.txt | head -n 1 | awk '{print $4}')
PARALLEL_OPS_TIME=$(grep "BenchmarkTensor_ParallelForEach/100x100" benchmark_results.txt | head -n 1 | awk '{print $4}')
LARGE_OPS_TIME=$(grep "BenchmarkNewTensor/shape_\[100_100\]" benchmark_results.txt | head -n 1 | awk '{print $4}')

# Generate PR description
cat << EOF > pr_description.md

## Test Coverage
- Current coverage: ${COVERAGE}
- Coverage changes: <previous> → ${COVERAGE}

## Performance Metrics
### Memory Usage
- Allocations per operation:
  - New tensor creation: ${NEW_TENSOR_ALLOCS} allocs/op
  - Get/Set operations: ${GET_SET_ALLOCS} allocs/op
  - Parallel operations: ${PARALLEL_ALLOCS} allocs/op

### CPU Performance
- Operation timing:
  - Basic operations: ${BASIC_OPS_TIME} ns/op
  - Parallel operations: ${PARALLEL_OPS_TIME} ns/op
  - Large tensor operations: ${LARGE_OPS_TIME} ns/op

EOF

echo "PR description generated in pr_description.md" 
