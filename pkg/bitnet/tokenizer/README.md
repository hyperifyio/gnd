# BitNet Tokenizer

This package implements the tokenization and detokenization functionality for the BitNet model.

## Components

### Tokenization
- Text to token ID conversion
- Subword tokenization
- Special token handling
- Context length management

### Detokenization
- Token ID to text conversion
- Special token filtering
- Output formatting
- Error handling

## Implementation Status

### Completed
- [x] Basic tokenization
- [x] Special token support
- [x] Context length validation
- [x] Error handling

### In Progress
- [ ] Performance optimization
  - [ ] Parallel tokenization
  - [ ] Memory usage optimization
  - [ ] Batch processing support
- [ ] Testing & Benchmarking
  - [ ] Tokenization accuracy tests
  - [ ] Performance benchmarks
  - [ ] Edge case handling

## Usage

```go
import "github.com/hyperifyio/gnd/pkg/bitnet/tokenizer"

// Create a new tokenizer
tok := tokenizer.NewTokenizer()

// Tokenize text
tokens, err := tok.Tokenize("Your input text")

// Detokenize tokens
text, err := tok.Detokenize(tokens)
```

## Features

- Support for BitNet's vocabulary
- Efficient tokenization algorithms
- Context length management (4096 tokens)
- Thread-safe operations

## Related Issues

- #170: Main feature implementation
- #190: Token decoding and inference loop
- #191: Parallelize with Goroutines
- #192: Testing & Performance Tuning 