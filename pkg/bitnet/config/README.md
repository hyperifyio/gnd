# BitNet Configuration

This package manages the configuration and constants used throughout the BitNet implementation.

## Components

### Runtime Configuration
- Model parameters and hyperparameters
- Performance tuning options
- Memory management settings
- Thread count configuration

### Constants
- Model architecture constants
- Quantization parameters
- Memory pool sizes
- Performance thresholds

## Implementation Status

### Completed
- [x] Basic configuration structure
- [x] Runtime parameters
- [x] Model constants
- [x] Memory settings

### In Progress
- [ ] Performance tuning (Issue #191)
  - [ ] Thread count optimization
    - [ ] CPU core detection
    - [ ] Dynamic thread allocation
  - [ ] Memory pool sizing
    - [ ] Target: ~0.4GB for 2B model
    - [ ] Efficient memory allocation
  - [ ] Batch size configuration
    - [ ] Optimal batch sizes
    - [ ] Memory-aware batching
- [ ] Testing & Benchmarking (Issue #192)
  - [ ] Configuration validation
  - [ ] Performance impact analysis
  - [ ] Memory usage verification

## Usage

```go
import "github.com/hyperifyio/gnd/pkg/bitnet/config"

// Create runtime configuration
cfg := config.NewRuntimeConfig()

// Configure thread count
cfg.SetThreadCount(runtime.NumCPU())

// Set memory pool size
cfg.SetMemoryPoolSize(1024 * 1024 * 1024) // 1GB
```

## Configuration Options

### Performance
- Thread count: Number of goroutines for parallel processing
  - Default: Number of CPU cores
  - Target: Optimize for 6x speedup on x86 CPUs
- Memory pool size: Size of the memory pool for tensor operations
  - Target: ~0.4GB for 2B model
- Batch size: Size of batches for processing
  - Configurable based on available memory

### Model
- Context length: Maximum number of tokens (4096)
- Quantization bits: 1.58-bit quantization
- Model dimensions: Hidden size, number of layers, etc.

## Related Issues

- #170: Main feature implementation
- #191: Parallelize with Goroutines
- #192: Testing & Performance Tuning 