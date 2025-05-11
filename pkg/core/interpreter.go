package core

// Interpreter defines the methods that an interpreter must implement
type Interpreter interface {
	// GetSlots returns the slots map
	GetSlots() map[string]interface{}

	// GetSubroutines returns the subroutines map
	GetSubroutines() map[string][]*Instruction

	// GetScriptDir returns the script directory
	GetScriptDir() string

	// GetLogIndent returns the current log indentation level
	GetLogIndent() int

	// SetLogIndent sets the log indentation level
	SetLogIndent(indent int)

	// LogDebug logs a debug message with proper indentation
	LogDebug(format string, args ...interface{})

	// ExecuteInstruction executes a single GND instruction and returns its result
	ExecuteInstruction(op *Instruction, idx int) (interface{}, error)
}

// NewFunc is a function type that creates a new core instance
type NewFunc func(scriptDir string) Interpreter
