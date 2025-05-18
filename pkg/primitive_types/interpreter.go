package primitive_types

import (
	"github.com/hyperifyio/gnd/pkg/parsers"
)

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

	// LogInfo logs a info message with proper indentation
	LogInfo(format string, args ...interface{})

	// LogWarn logs a warning message with proper indentation
	LogWarn(format string, args ...interface{})

	// LogError logs a error message with proper indentation
	LogError(format string, args ...interface{})

	// ExecuteInstructionBlock executes a sequence of instructions and returns the last result
	// source is the source of the instructions (for debug purporeses only), e.g., a file name or a source identificating string
	ExecuteInstructionBlock(source string, input interface{}, instructions []*parsers.Instruction) (interface{}, error)

	// SetSlot sets a slot value
	SetSlot(name string, value interface{}) error

	// GetSlot gets a slot value
	GetSlot(name string) (interface{}, error)

	// GetSubroutineInstructions retrieves the instructions for a subroutine
	GetSubroutineInstructions(path string) ([]*parsers.Instruction, error)

	// ResolveOpcode
	ResolveOpcode(opcode string) string

	// NewInterpreterWithParent
	NewInterpreterWithParent(scriptDir string, initialSlots map[string]interface{}) Interpreter
}
