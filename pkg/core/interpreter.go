package core

import "github.com/hyperifyio/gnd/pkg/parsers"

// Interpreter defines the methods that an interpreter must implement
type Interpreter interface {
	// GetScriptDir returns the script directory
	GetScriptDir() string

	// GetLogIndent returns the current log indentation level
	GetLogIndent() int

	// SetLogIndent sets the log indentation level
	SetLogIndent(indent int)

	// LogDebug logs a debug message with proper indentation
	LogDebug(format string, args ...interface{})

	// ExecuteInstruction executes a single GND instruction and returns its result
	ExecuteInstruction(opcode string, destination *parsers.PropertyRef, arguments []interface{}) (interface{}, error)

	// ExecuteInstructionBlock executes a sequence of instructions and returns the last result
	// source is the source of the instructions (for debug purporeses only), e.g., a file name or a source identificating string
	ExecuteInstructionBlock(source string, input interface{}, instructions []*parsers.Instruction) (interface{}, error)

	// SetSlot sets a slot value
	SetSlot(name string, value interface{}) error

	// GetSlot gets a slot value
	GetSlot(name string) (interface{}, error)
}
