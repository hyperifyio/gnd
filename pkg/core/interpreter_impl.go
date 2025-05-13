package core

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive"
	"github.com/hyperifyio/gnd/pkg/units"
)

// InterpreterImpl represents the execution environment
type InterpreterImpl struct {
	Slots       map[string]interface{}
	Subroutines map[string][]*Instruction
	ScriptDir   string // Directory of the currently executing script
	LogIndent   int    // Current log indentation level
	UnitsFS     fs.FS  // Embedded filesystem containing GND units
}

// NewInterpreter creates a new core instance
var NewInterpreter NewFunc = func(scriptDir string) Interpreter {
	return &InterpreterImpl{
		Slots:       make(map[string]interface{}),
		Subroutines: make(map[string][]*Instruction),
		ScriptDir:   scriptDir,
		LogIndent:   0,
		UnitsFS:     units.GetUnitsFS(),
	}
}

// NewInterpreterWithSlots creates a new interpreter with initial slots
func NewInterpreterWithSlots(scriptDir string, initialSlots map[string]interface{}) Interpreter {
	return &InterpreterImpl{
		Slots:       initialSlots,
		Subroutines: make(map[string][]*Instruction),
		ScriptDir:   scriptDir,
		LogIndent:   0,
		UnitsFS:     units.GetUnitsFS(),
	}
}

// GetScriptDir returns the script directory
func (i *InterpreterImpl) GetScriptDir() string {
	return i.ScriptDir
}

// GetLogIndent returns the current log indentation level
func (i *InterpreterImpl) GetLogIndent() int {
	return i.LogIndent
}

// SetLogIndent sets the log indentation level
func (i *InterpreterImpl) SetLogIndent(indent int) {
	i.LogIndent = indent
}

// getLogPrefix returns the current log prefix based on indentation
func (i *InterpreterImpl) getLogPrefix() string {
	if i.LogIndent == 0 {
		return ""
	}
	return strings.Repeat("  ", i.LogIndent)
}

// LogDebug logs a debug message with proper indentation
func (i *InterpreterImpl) LogDebug(format string, args ...interface{}) {
	prefix := i.getLogPrefix()
	log.Printf(log.Debug, prefix+format, args...)
}

// loadSubroutine loads a subroutine from a file
func (i *InterpreterImpl) loadSubroutine(name string) error {
	// Construct the path to the subroutine file
	subPath := filepath.Join(i.GetScriptDir(), name+".gnd")

	// Parse the subroutine file
	instructions, err := ParseFile(subPath)
	if err != nil {
		return fmt.Errorf("failed to parse subroutine %s: %v", name, err)
	}

	// Store the subroutine
	i.Subroutines[name] = instructions
	return nil
}

// ExecuteInstructions executes a sequence of instructions and returns the last result
func (i *InterpreterImpl) ExecuteInstructions(instructions []*Instruction) (interface{}, error) {
	var lastResult interface{}
	for idx, op := range instructions {
		result, err := i.ExecuteInstruction(op, idx)
		if err != nil {
			return nil, fmt.Errorf("failed at instruction %d: %v", idx, err)
		}
		lastResult = result
	}
	return lastResult, nil
}

// executeSubroutine executes a subroutine with the given arguments
func (i *InterpreterImpl) executeSubroutine(name string, args []interface{}) (interface{}, error) {
	// Check if the subroutine is already loaded
	instructions, ok := i.Subroutines[name]
	if !ok {
		// Try to load the subroutine
		if err := i.loadSubroutine(name); err != nil {
			return nil, err
		}
		instructions = i.Subroutines[name]
	}

	// Create a new scope for the subroutine with just the arguments
	subInterpreter := NewInterpreterWithSlots(i.GetScriptDir(), map[string]interface{}{
		"_": args,
	})

	// Execute the instructions using the new method
	result, err := subInterpreter.ExecuteInstructions(instructions)
	if err != nil {
		return nil, fmt.Errorf("subroutine %s failed: %v", name, err)
	}

	return result, nil
}

// ExecuteInstruction executes a single GND instruction and returns its result
func (i *InterpreterImpl) ExecuteInstruction(op *Instruction, idx int) (interface{}, error) {
	if op == nil {
		return nil, nil
	}

	// Handle subroutine calls
	if op.IsSubroutine {
		return i.ExecuteSubroutineCall(op)
	}

	// Log regular instruction
	i.LogDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

	// Resolve arguments by mapping context properties
	resolvedArgs, err := parsers.MapContextProperties(i.Slots, op.Arguments)
	if err != nil {
		return nil, err
	}

	// If no arguments provided, use current value of _
	if len(resolvedArgs) == 0 {
		if val, ok := i.Slots["_"]; ok {
			i.LogDebug("Using current value of _ (%v) as argument", val)
			resolvedArgs = []interface{}{val}
		} else {
			i.LogDebug("No value in _, using empty string as argument")
			resolvedArgs = []interface{}{""}
		}
	}

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return nil, fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		return nil, fmt.Errorf("%s: %v", op.Opcode, err)
	}

	i.LogDebug("primitive result: %v", result)

	// Check if this is an ExitResult
	if exitResult, ok := GetExitResult(result); ok {
		i.LogDebug("exit result detected with code %d", exitResult.Code)
		// Store the value in the destination slot before exiting
		if exitResult.Value != nil {
			i.LogDebug("storing value %v (type: %T) in destination %s", exitResult.Value, exitResult.Value, op.Destination)
			i.Slots[op.Destination] = exitResult.Value
			i.LogDebug("after storing, destination %s contains: %v (type: %T)", op.Destination, i.Slots[op.Destination], i.Slots[op.Destination])
			return exitResult.Value, &ExitErrorWithValue{
				Code:  exitResult.Code,
				Value: exitResult.Value,
			}
		}
		return nil, &ExitError{Code: exitResult.Code}
	}

	// For debug output, escape newlines to make them visible
	i.LogDebug("%s = %s", op.Destination, log.StringifyValue(result))

	// Store the result in the destination slot
	i.LogDebug("storing result %v in destination %s", result, op.Destination)
	i.Slots[op.Destination] = result
	return result, nil
}

// ExecuteSubroutineCall handles the complete flow of executing a subroutine call,
// including argument resolution and result storage
func (i *InterpreterImpl) ExecuteSubroutineCall(op *Instruction) (interface{}, error) {
	// Extract the base name without .gnd extension
	baseName := strings.TrimSuffix(op.SubroutinePath, ".gnd")
	i.LogDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

	// Resolve arguments by mapping context properties
	resolvedArgs, err := parsers.MapContextProperties(i.Slots, op.Arguments)
	if err != nil {
		return nil, err
	}

	// Execute the subroutine with the resolved arguments
	result, err := i.executeSubroutine(baseName, resolvedArgs)
	if err != nil {
		return nil, err
	}

	// Store the result in the destination slot
	i.LogDebug("Storing subroutine result %v in destination %s", result, op.Destination)
	i.Slots[op.Destination] = result
	return result, nil
}
