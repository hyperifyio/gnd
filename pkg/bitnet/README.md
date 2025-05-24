# BitNet Go Implementation

This package implements Microsoft's BitNet b1.58-2B-4T model in pure Go, focusing on inference-only functionality. The implementation is designed to be performant on CPU using goroutine-based concurrency.

## Package Structure

```
bitnet/
├── internal/
│   ├── config/      # Configuration and constants
│   ├── math/        # Pure Go math operations
│   └── utils/       # Utility functions
├── model/           # Model structures and interfaces
├── quantization/    # 1.58-bit quantization implementation
└── tensor/          # Tensor operations
```

## Features

- Pure Go implementation (no CGo or external C/C++ dependencies)
- Multi-core CPU utilization through goroutines
- 4096-token context support
- 1.58-bit quantization
- Memory-efficient tensor operations

## Usage

```go
import "github.com/hyperifyio/gnd/pkg/bitnet"

// Initialize the model
config := bitnet.NewRuntimeConfig()
model := bitnet.NewModel(config)

// Run inference
result, err := model.Infer("Your input text here")
```

## Development Status

This is a work in progress. Current implementation status:
- [x] Project setup and basic structure
- [x] Model weights and tokenizer integration
  - [x] Model file loading with memory pooling
  - [x] Efficient chunk-based reading
  - [x] Performance benchmarks
- [ ] Core tensor operations
- [ ] Quantization implementation
- [ ] Model inference
- [ ] Performance optimization

## License

See the main project license. 