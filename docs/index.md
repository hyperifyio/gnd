# Gendo Documentation

Gendo is a locally executed, AI-assisted programming system designed for offline code generation and execution. It provides a minimal, deterministic format that enables AI to generate, analyze, and repair code while remaining human-readable and maintainable.

## Core Components

The Gendo toolchain consists of three main executables:

- **`gndc`**: The compiler front-end that processes natural language headers and generates implementation files
- **`gnd`**: The runtime interpreter for executing `.gnd` scripts
- **`gndtest`**: The test runner for evaluating test cases

## Language Documentation

### Core Syntax
- [Gendo Syntax Specification](gnd-syntax.md) - Complete language syntax and file format specification

### Built-in Operations

#### Flow Control
- [let](let-syntax.md) - Variable assignment and binding
- [return](return-syntax.md) - Early return from execution
- [exit](exit-syntax.md) - Program termination

#### String Operations
- [concat](concat-syntax.md) - String concatenation
- [trim](trim-syntax.md) - String trimming
- [uppercase](uppercase-syntax.md) - Convert to uppercase
- [lowercase](lowercase-syntax.md) - Convert to lowercase

#### Output and Logging
- [print](print-syntax.md) - Standard output
- [log](log-syntax.md) - Logging
- [debug](debug-syntax.md) - Debug output
- [info](info-syntax.md) - Information messages
- [warn](warn-syntax.md) - Warning messages
- [error](error-syntax.md) - Error messages

#### AI Integration
- [prompt](prompt-syntax.md) - AI prompt handling

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
