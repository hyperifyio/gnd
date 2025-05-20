#!/bin/bash

# Generate test coverage report
echo "Generating test coverage report..."
go test ./pkg/bitnet/... -coverprofile=coverage.out
COVERAGE=$(go tool cover -func=coverage.out | grep total | awk '{print $3}')

# Run benchmarks
echo "Running benchmarks..."
./scripts/run_benchmarks.sh > benchmark_results.txt

# Extract tensor benchmark results
NEW_TENSOR_ALLOCS=$(grep "BenchmarkNewTensor/shape_\[100\]" benchmark_results.txt | head -n 1 | awk '{print $5}')
GET_SET_ALLOCS=$(grep "BenchmarkTensor_Get/2D_access" benchmark_results.txt | head -n 1 | awk '{print $5}')
PARALLEL_ALLOCS=$(grep "BenchmarkTensor_ParallelForEach/100x100" benchmark_results.txt | head -n 1 | awk '{print $5}')

BASIC_OPS_TIME=$(grep "BenchmarkTensor_Get/2D_access" benchmark_results.txt | head -n 1 | awk '{print $4}')
PARALLEL_OPS_TIME=$(grep "BenchmarkTensor_ParallelForEach/100x100" benchmark_results.txt | head -n 1 | awk '{print $4}')
LARGE_OPS_TIME=$(grep "BenchmarkNewTensor/shape_\[100_100\]" benchmark_results.txt | head -n 1 | awk '{print $4}')

# Extract BitNet model benchmark results
MODEL_LOAD_TIME=$(grep "BenchmarkModel_LoadWeights" benchmark_results.txt | head -n 1 | awk '{print $4}')
MODEL_LOAD_ALLOCS=$(grep "BenchmarkModel_LoadWeights" benchmark_results.txt | head -n 1 | awk '{print $5}')
MODEL_INFER_TIME=$(grep "BenchmarkModel_Infer" benchmark_results.txt | head -n 1 | awk '{print $4}')
MODEL_INFER_ALLOCS=$(grep "BenchmarkModel_Infer" benchmark_results.txt | head -n 1 | awk '{print $5}')
TERNARY_WEIGHTS_TIME=$(grep "BenchmarkModel_ReadTernaryWeights" benchmark_results.txt | head -n 1 | awk '{print $4}')
TERNARY_WEIGHTS_ALLOCS=$(grep "BenchmarkModel_ReadTernaryWeights" benchmark_results.txt | head -n 1 | awk '{print $5}')

# Generate PR description
cat << EOF > pr_description.md
## Changes
- [ ] List of specific changes made
- [ ] Include file paths and line numbers for major changes
- [ ] Reference related issues/tickets

## Test Coverage
- Current coverage: ${COVERAGE}
- Coverage changes: <previous> → ${COVERAGE}

## Performance Metrics
### Memory Usage
#### Tensor Operations
- Allocations per operation:
  - New tensor creation: ${NEW_TENSOR_ALLOCS} allocs/op
  - Get/Set operations: ${GET_SET_ALLOCS} allocs/op
  - Parallel operations: ${PARALLEL_ALLOCS} allocs/op

#### BitNet Model Operations
- Allocations per operation:
  - Model weights loading: ${MODEL_LOAD_ALLOCS} allocs/op
  - Model inference: ${MODEL_INFER_ALLOCS} allocs/op
  - Ternary weights reading: ${TERNARY_WEIGHTS_ALLOCS} allocs/op

### CPU Performance
#### Tensor Operations
- Operation timing:
  - Basic operations: ${BASIC_OPS_TIME} ns/op
  - Parallel operations: ${PARALLEL_OPS_TIME} ns/op
  - Large tensor operations: ${LARGE_OPS_TIME} ns/op

#### BitNet Model Operations
- Operation timing:
  - Model weights loading: ${MODEL_LOAD_TIME} ns/op
  - Model inference: ${MODEL_INFER_TIME} ns/op
  - Ternary weights reading: ${TERNARY_WEIGHTS_TIME} ns/op

## Areas for Improvement
### High Priority
- [ ] Add tests for internal packages
- [ ] Optimize memory allocations in model operations
- [ ] Implement proper tokenization (TODO #174)
- [ ] Implement proper self-attention (TODO #186)

### Medium Priority
- [ ] Improve error handling in model operations
- [ ] Add more comprehensive benchmarks
- [ ] Enhance documentation
- [ ] Implement proper feed-forward network (TODO #187)

### Low Priority
- [ ] Consider SIMD optimizations
- [ ] Add more model operations
- [ ] Improve test organization
- [ ] Implement proper output generation (TODO #189)
EOF

echo "PR description generated in pr_description.md" 
