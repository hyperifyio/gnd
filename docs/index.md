# Gendo Documentation

Gendo is a locally executed, AI-assisted programming system designed for offline code generation and execution. It provides a minimal, deterministic format that enables AI to generate, analyze, and repair code while remaining human-readable and maintainable.

## Core Components

The Gendo toolchain consists of three main executables:

- **`gnd`**: The runtime interpreter for executing `.gnd` scripts ([Roadmap](https://github.com/hyperifyio/gnd/issues/31))
- **`gndc`**: The compiler front-end that processes natural language headers and generates implementation files ([Roadmap](https://github.com/hyperifyio/gnd/issues/24))
- **`gndtest`**: The test runner for evaluating test cases ([Roadmap](https://github.com/hyperifyio/gnd/issues/23))

## Language Documentation

### Core Syntax
- [Gendo Syntax Specification](gnd-syntax.md) - Complete language syntax and file format specification

### Built-in Operations

#### Flow Control
- [let](let-syntax.md) - Variable assignment and binding
- [return](return-syntax.md) - Early return from execution
- [exit](exit-syntax.md) - Program termination
- [exec](exec-syntax.md) - Execute instructions
- [compile](compile-syntax.md) - Compile instructions
- [code](code-syntax.md) - Code block handling
- [async](async-syntax.md) - Asynchronous execution
- [await](await-syntax.md) - Wait for async operations
- [wait](wait-syntax.md) - Wait for conditions
- [throw](throw-syntax.md) - Error handling
- [status](status-syntax.md) - Status checking

#### String Operations
- [concat](concat-syntax.md) - String concatenation
- [trim](trim-syntax.md) - String trimming
- [uppercase](uppercase-syntax.md) - Convert to uppercase
- [lowercase](lowercase-syntax.md) - Convert to lowercase
- [eq](eq-syntax.md) - Equality comparison

#### Output and Logging
- [print](print-syntax.md) - Standard output
- [log](log-syntax.md) - Logging
- [debug](debug-syntax.md) - Debug output
- [info](info-syntax.md) - Information messages
- [warn](warn-syntax.md) - Warning messages
- [error](error-syntax.md) - Error messages

#### AI Integration
- [prompt](prompt-syntax.md) - AI prompt handling

#### Type Operations
- [string](string-syntax.md) - String type operations
- [bool](bool-syntax.md) - Boolean type operations
- [int](int-syntax.md) - Integer type operations
- [int8](int8-syntax.md) - 8-bit integer operations
- [int16](int16-syntax.md) - 16-bit integer operations
- [int32](int32-syntax.md) - 32-bit integer operations
- [int64](int64-syntax.md) - 64-bit integer operations
- [uint](uint-syntax.md) - Unsigned integer operations
- [uint8](uint8-syntax.md) - 8-bit unsigned integer operations
- [uint16](uint16-syntax.md) - 16-bit unsigned integer operations
- [uint32](uint32-syntax.md) - 32-bit unsigned integer operations
- [uint64](uint64-syntax.md) - 64-bit unsigned integer operations
- [float32](float32-syntax.md) - 32-bit floating point operations
- [float64](float64-syntax.md) - 64-bit floating point operations

## Getting Started

To begin using Gendo:

1. Write your intent in a `.llm` file
2. Optionally provide implementation guidance in a `.gnd.llm` file
3. Use `gndc` to generate the implementation
4. Execute the resulting `.gnd` file with `gnd`

For testing, write your test cases in `.test.gnd.llm` or `.test.llm` files and use `gndtest` to run them.

## Key Features

- **Offline Operation**: All phases run locally with no hidden state
- **Deterministic**: Consistent behavior across executions
- **AI-Friendly**: Designed for automated code generation and analysis
- **Human-Readable**: Simple syntax for manual inspection and modification
- **Extensible**: Modular design supporting future additions

## File Organization

A Gendo unit consists of multiple files sharing the same base name:
- `.llm` - Natural language header
- `.gnd.llm` - Optional implementation prompt
- `.gnd` - Generated implementation
- `.gnc` - Compact compiled form (optional)

Numbered fragments (e.g., `010-foo.gnd`, `020-foo.gnd`) are automatically concatenated in order.
