package core

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/hyperifyio/gnd/pkg/parsers"

	"github.com/hyperifyio/gnd/pkg/log"
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

// GetSlots returns the slots map
func (i *InterpreterImpl) GetSlots() map[string]interface{} {
	return i.Slots
}

// GetSubroutines returns the subroutines map
func (i *InterpreterImpl) GetSubroutines() map[string][]*Instruction {
	return i.Subroutines
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

// executeSubroutine executes a subroutine
func (i *InterpreterImpl) executeSubroutine(name string) error {
	// Check if the subroutine is already loaded
	instructions, ok := i.Subroutines[name]
	if !ok {
		// Try to load the subroutine
		if err := i.loadSubroutine(name); err != nil {
			return err
		}
		instructions = i.Subroutines[name]
	}

	// Create a new core for the subroutine
	subInterpreter := NewInterpreter(i.GetScriptDir())

	// Copy the slots from the parent core
	for k, v := range i.Slots {
		subInterpreter.GetSlots()[k] = v
	}

	// Execute each instruction
	for idx, op := range instructions {
		if err := subInterpreter.ExecuteInstruction(op, idx); err != nil {
			return fmt.Errorf("subroutine %s failed at instruction %d: %v", name, idx, err)
		}
	}

	// Copy the slots back to the parent core
	for k, v := range subInterpreter.GetSlots() {
		i.Slots[k] = v
	}

	return nil
}

// ExecuteInstruction executes a single GND instruction
func (i *InterpreterImpl) ExecuteInstruction(op *Instruction, idx int) error {
	if op == nil {
		return nil
	}

	// Handle subroutine calls
	if op.IsSubroutine {
		// Extract the base name without .gnd extension
		baseName := strings.TrimSuffix(op.SubroutinePath, ".gnd")
		i.LogDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

		// Resolve arguments by mapping context properties
		resolvedArgs, err := parsers.MapContextProperties(i.Slots, op.Arguments)
		if err != nil {
			return err
		}

		// Store the resolved arguments in _
		i.LogDebug("Storing resolved args %v in _ before subroutine call", resolvedArgs)
		i.Slots["_"] = resolvedArgs

		// Execute the subroutine
		err = i.executeSubroutine(baseName)
		if err != nil {
			return err
		}

		// Store the result in the destination slot
		if val, ok := i.Slots["_"]; ok {
			i.LogDebug("Storing subroutine result %v in destination %s", val, op.Destination)
			i.Slots[op.Destination] = val
		} else {
			i.LogDebug("No result in _ to store in destination %s", op.Destination)
		}

		return nil
	}

	// Log regular instruction
	i.LogDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

	// Resolve arguments by mapping context properties
	resolvedArgs, err := parsers.MapContextProperties(i.Slots, op.Arguments)
	if err != nil {
		return err
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

	// For return primitive, handle unquoted strings
	if op.Opcode == "/gnd/return" {
		// Check if all arguments are unquoted strings
		allUnquoted := true
		strArgs := make([]string, 0, len(op.Arguments))
		for _, arg := range op.Arguments {
			if strArg, ok := arg.(string); ok {
				if len(strArg) >= 2 && strArg[0] == '"' && strArg[len(strArg)-1] == '"' {
					allUnquoted = false
					break
				}
				strArgs = append(strArgs, strArg)
			} else {
				allUnquoted = false
				break
			}
		}
		if allUnquoted && len(strArgs) > 0 {
			i.LogDebug("Joining unquoted strings with spaces: %v", strArgs)
			resolvedArgs = []interface{}{strings.Join(strArgs, " ")}
		}
	}

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		return fmt.Errorf("%s: %v", op.Opcode, err)
	}

	i.LogDebug("primitive result: %v", result)

	// Check if this is a return with exit signal
	if resultMap, ok := result.(map[string]interface{}); ok {
		i.LogDebug("result is a map: %v", resultMap)
		if exit, ok := resultMap["exit"].(bool); ok && exit {
			i.LogDebug("exit signal detected")
			// Get the exit code if provided
			exitCode := 0
			if code, ok := resultMap["code"].(int); ok {
				exitCode = code
			}
			// Store the value in the destination slot before exiting
			if val, ok := resultMap["value"]; ok {
				dest := op.Destination
				if d, ok := resultMap["destination"].(string); ok {
					dest = d
				}
				i.LogDebug("storing value %v (type: %T) in destination %s", val, val, dest)
				i.Slots[dest] = val
				i.LogDebug("after storing, destination %s contains: %v (type: %T)", dest, i.Slots[dest], i.Slots[dest])
				// Print all values
				switch v := val.(type) {
				case []interface{}:
					i.LogDebug("printing array of values: %v (type: %T)", v, v)
					for _, item := range v {
						i.LogDebug("printing array item: %v (type: %T)", item, item)
						fmt.Print(item)
					}
				default:
					i.LogDebug("printing single value: %v (type: %T)", v, v)
					fmt.Print(val)
				}
				os.Stdout.Sync()
				return &ExitError{Code: exitCode}
			} else {
				i.LogDebug("no value found in result map")
				return &ExitError{Code: exitCode}
			}
		}
	}

	// For debug output, escape newlines to make them visible
	debugResult := fmt.Sprintf("%v", result)
	debugResult = strings.ReplaceAll(debugResult, "\n", "\\n")
	i.LogDebug("%s = %s", op.Destination, debugResult)

	// Store the result in the destination slot
	i.LogDebug("storing result %v in destination %s", result, op.Destination)
	i.Slots[op.Destination] = result
	return nil
}
