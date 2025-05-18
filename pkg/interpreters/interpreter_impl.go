package interpreters

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	primitive_types2 "github.com/hyperifyio/gnd/pkg/primitive_types"
	"github.com/hyperifyio/gnd/pkg/primitives"
	"io/fs"
	"os"
	"strings"

	"github.com/hyperifyio/gnd/pkg/helpers"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/units"
)

// InterpreterImpl represents the execution environment
type InterpreterImpl struct {
	Slots       map[string]interface{}
	Subroutines map[string][]*parsers.Instruction
	ScriptDir   string                       // Directory of the currently executing script
	LogIndent   int                          // Current log indentation level
	UnitsFS     fs.FS                        // Embedded filesystem containing GND units
	OpcodeMap   map[string]string            // Map of opcode aliases
	parent      primitive_types2.Interpreter // Parent interpreter for nested calls
}

// NewInterpreter creates a new core instance
func NewInterpreter(
	scriptDir string,
	opcodeMap map[string]string,
) primitive_types2.Interpreter {
	return &InterpreterImpl{
		Slots:       make(map[string]interface{}),
		Subroutines: make(map[string][]*parsers.Instruction),
		ScriptDir:   scriptDir,
		LogIndent:   0,
		UnitsFS:     units.GetUnitsFS(),
		OpcodeMap:   opcodeMap,
	}
}

// NewInterpreterWithParent creates a new interpreter with initial slots and opcode aliases
func NewInterpreterWithParent(
	scriptDir string,
	initialSlots map[string]interface{},
	parent primitive_types2.Interpreter,
) primitive_types2.Interpreter {
	return &InterpreterImpl{
		Slots:       initialSlots,
		Subroutines: make(map[string][]*parsers.Instruction),
		ScriptDir:   scriptDir,
		LogIndent:   0,
		UnitsFS:     units.GetUnitsFS(),
		OpcodeMap:   make(map[string]string),
		parent:      parent,
	}
}

// SetSlot sets a slot value
func (i *InterpreterImpl) SetSlot(name string, value interface{}) error {
	if name == "" {
		return fmt.Errorf("SetSlot: empty slot name")
	}
	i.Slots[name] = value
	return nil
}

// GetSlot gets a slot value
func (i *InterpreterImpl) GetSlot(name string) (interface{}, error) {
	if name == "" {
		return nil, fmt.Errorf("GetSlot: empty slot name")
	}
	value, ok := i.Slots[name]
	if !ok {
		return nil, fmt.Errorf("GetSlot: slot not found: %s", name)
	}
	return value, nil
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

// GetLogPrefix returns the current log prefix based on indentation
func (i *InterpreterImpl) GetLogPrefix() string {
	if i.LogIndent == 0 {
		return ""
	}
	return strings.Repeat("  ", i.LogIndent)
}

// LogDebug logs a debug message with proper indentation
func (i *InterpreterImpl) LogDebug(format string, args ...interface{}) {
	prefix := i.GetLogPrefix()
	loggers.Printf(loggers.Debug, prefix+format, args...)
}

// LogInfo logs a Info message with proper indentation
func (i *InterpreterImpl) LogInfo(format string, args ...interface{}) {
	prefix := i.GetLogPrefix()
	loggers.Printf(loggers.Info, prefix+format, args...)
}

// LogWarn logs a Warn message with proper indentation
func (i *InterpreterImpl) LogWarn(format string, args ...interface{}) {
	prefix := i.GetLogPrefix()
	loggers.Printf(loggers.Warn, prefix+format, args...)
}

// LogError logs a Error message with proper indentation
func (i *InterpreterImpl) LogError(format string, args ...interface{}) {
	prefix := i.GetLogPrefix()
	loggers.Printf(loggers.Error, prefix+format, args...)
}

// LoadSubroutine loads a subroutine from a file
func (i *InterpreterImpl) LoadSubroutine(subPath string) error {

	var content []byte

	if strings.HasPrefix(subPath, "/gnd/") {
		// Parse subroutine file from embedded filesystem i.UnitsFS
		c, err := fs.ReadFile(i.UnitsFS, subPath[1:])
		if err != nil {
			return fmt.Errorf("[%s]: LoadSubroutine: failed to read embedded subroutine:\n  %v", subPath, err)
		}
		content = c
	} else {
		// Read the subroutine file from filesystem
		c, err := os.ReadFile(subPath)
		if err != nil {
			return fmt.Errorf("[%s]: LoadSubroutine: failed to read subroutine:\n  %v", subPath, err)
		}
		content = c
	}

	// Parse the subroutine file
	instructions, err := parsers.ParseInstructionLines(subPath, string(content))
	if err != nil {
		return fmt.Errorf("[%s]: LoadSubroutine: failed to parse subroutine:\n  %v", subPath, err)
	}

	// Store the subroutine
	i.Subroutines[subPath] = instructions
	return nil
}

// ExecuteInstructionBlock executes a sequence of instructions and returns the last result
func (i *InterpreterImpl) ExecuteInstructionBlock(source string, input interface{}, instructions []*parsers.Instruction) (interface{}, error) {
	lastResult := input
	for idx, op := range instructions {
		if op != nil {
			i.LogDebug("[%s:%d]: ExecuteInstructionBlock: %v <- %s %v", source, idx, op.Destination, op.Opcode, op.Arguments)

			// Check if the opcode exists in the default alias map
			var opcode = i.ResolveOpcode(op.Opcode)
			var arguments = op.Arguments
			var destination = op.Destination

			// Resolve arguments by mapping context properties
			resolvedArgs, err := i.LoadArguments(opcode, arguments)
			if err != nil {
				i.LogDebug("[%s]: ExecuteInstructionBlock: argument parsing failed: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)
				return nil, fmt.Errorf("[%s]: ExecuteInstructionBlock: failed to load arguments: %s", opcode, err)
			}
			i.LogDebug("[%s]: ExecuteInstructionBlock: Resolved arguments as: %v from %v", opcode, resolvedArgs, arguments)

			var result interface{}
			prim, ok := primitive_services.GetPrimitive(opcode)
			if !ok {
				i.LogDebug("[%s]: ExecuteInstructionBlock: subroutine: %v <- %s %v", opcode, destination, opcode, resolvedArgs)
				result, err = i.ExecuteSubroutineCall(opcode, destination, resolvedArgs)

				if err != nil {
					i.LogDebug("[%s]: ExecuteInstructionBlock: subroutine had error: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)
					return nil, fmt.Errorf("\n  %s:%d: %v", source, idx, err)
				}

			} else {
				i.LogDebug("[%s]: ExecuteInstructionBlock: primitive: %v <- %s %v", opcode, destination, opcode, resolvedArgs)

				result, err = prim.Execute(resolvedArgs)

				if err != nil {

					i.LogDebug("[%s]: ExecuteInstructionBlock: primitive had error: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)

					var ok2 bool
					var handler primitive_types2.BlockErrorResultHandler
					if handler, ok2 = prim.(primitive_types2.BlockErrorResultHandler); ok2 {

						i.LogDebug("[%s]: ExecuteInstructionBlock: handle error using BlockErrorResultHandler: %v <- %s %v", opcode, destination, opcode, resolvedArgs)
						result, err = handler.HandleBlockErrorResult(err, i, destination, instructions)
						if err != nil {
							i.LogDebug("[%s]: ExecuteInstructionBlock: handler had error: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)
							return nil, fmt.Errorf("\n  %s:%d: %v", source, idx, err)
						}

						if returnValue, ok2 := primitives.GetReturnValue(result); ok2 {
							i.LogDebug("[%s]: return value detected: %v", source, returnValue.Value)
							return returnValue.Value, nil
						}

						i.LogDebug("[%s]: ExecuteInstructionBlock: handler provided result: %v <- %s %v: %v", opcode, destination, opcode, resolvedArgs, result)

					} else {
						i.LogDebug("[%s]: ExecuteInstructionBlock: no handler detected: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)
						return nil, fmt.Errorf("\n  %s:%d: %v", source, idx, err)
					}

				} else {

					var handler primitive_types2.BlockSuccessResultHandler
					if handler, ok = prim.(primitive_types2.BlockSuccessResultHandler); ok {
						i.LogDebug("[%s]: ExecuteInstructionBlock: handle result using BlockSuccessResultHandler: %v <- %s %v: %v", opcode, destination, opcode, resolvedArgs, result)
						result, err = handler.HandleBlockSuccessResult(result, i, destination, instructions)
						if err != nil {
							i.LogDebug("[%s]: ExecuteInstructionBlock: BlockSuccessResultHandler failed: %v <- %s %v: error: %v", opcode, destination, opcode, resolvedArgs, err)
							return nil, fmt.Errorf("\n  %s:%d: %v", source, idx, err)
						}
						i.LogDebug("[%s]: ExecuteInstructionBlock: we got result: %v <- %s %v: %v", opcode, destination, opcode, resolvedArgs, result)
					} else {
						// Store the result in the destination slot
						i.LogDebug("[%s]: ExecuteInstructionBlock: %v <- %v", prim.Name(), destination, result)
						i.Slots[destination.Name] = result
					}

				}

			}

			lastResult = result
		}
	}

	i.LogDebug("[%s]: ExecuteInstructionBlock: return by loop end: %v", source, lastResult)
	return lastResult, nil
}

// GetSubroutineInstructions retrieves the cached instructions for a subroutine
func (i *InterpreterImpl) GetSubroutineInstructions(path string) ([]*parsers.Instruction, error) {
	instructions, ok := i.Subroutines[path]
	if !ok {
		if err := i.LoadSubroutine(path); err != nil {
			return nil, fmt.Errorf("[%s]: GetSubroutineInstructions: loading failed: %v", path, err)
		}
		instructions, ok = i.Subroutines[path]
		if !ok {
			return nil, fmt.Errorf("[%s]: GetSubroutineInstructions: failed to find instructions", path)
		}
	}
	return instructions, nil
}

// ExecuteSubroutine executes a subroutine with the given arguments
func (i *InterpreterImpl) ExecuteSubroutine(name string, args []interface{}) (interface{}, error) {

	pwd := i.GetScriptDir()
	i.LogDebug("[%s]: ExecuteSubroutine: %v with pwd %s", name, args, pwd)

	// Check if the subroutine is already loaded
	subPath := helpers.SubroutinePath(name, pwd)
	i.LogDebug("[%s]: ExecuteSubroutine: subPath = %v", name, subPath)

	instructions, err := i.GetSubroutineInstructions(subPath)
	if err != nil {
		return nil, fmt.Errorf("[%s]: ExecuteSubroutine: loading failed: %v", name, err)
	}
	i.LogDebug("[%s]: Loaded %d instructions", name, len(instructions))

	// Create a new scope for the subroutine with just the arguments
	subInterpreter := NewInterpreterWithParent(
		pwd,
		map[string]interface{}{
			"_": args,
		},
		i,
	)

	// Execute the instructions using the new method
	result, err2 := subInterpreter.ExecuteInstructionBlock(subPath, args, instructions)
	if err2 != nil {
		return nil, fmt.Errorf("[%s]: ExecuteSubroutine: execute failed: %v", name, err2)
	}
	i.LogDebug("[%s]: result: %v", name, result)
	return result, nil
}

// ResolveOpcode resolves the opcode to its mapped value
func (i *InterpreterImpl) ResolveOpcode(opcode string) string {
	if mapped, exists := i.OpcodeMap[opcode]; exists {
		i.LogDebug("[%s]: Resolved as: %s", opcode, mapped)
		return mapped
	}
	if i.parent != nil {
		p := i.parent.ResolveOpcode(opcode)
		i.LogDebug("[%s]: Resolved from parent as: %s", opcode, p)
		return p
	}
	return opcode
}

// LoadArguments resolves arguments by mapping context properties
func (i *InterpreterImpl) LoadArguments(source string, arguments []interface{}) ([]interface{}, error) {

	// Resolve arguments by mapping context properties
	resolvedArgs, err := parsers.MapContextProperties(source, i.Slots, arguments)
	if err != nil {
		return nil, err
	}

	i.LogDebug("[%s]: LoadArguments: %v from %v", source, resolvedArgs, arguments)
	return resolvedArgs, nil
}

// ExecuteSubroutineCall handles the complete flow of executing a subroutine call,
// including argument resolution and result storage
func (i *InterpreterImpl) ExecuteSubroutineCall(opcode string, destination *parsers.PropertyRef, arguments []interface{}) (interface{}, error) {

	// Extract the base name without .gnd extension
	i.LogDebug("[%s]: ExecuteSubroutineCall: %v <- %s %v", opcode, destination, opcode, arguments)

	// Execute the subroutine with the resolved arguments
	result, err := i.ExecuteSubroutine(opcode, arguments)
	if err != nil {
		return nil, fmt.Errorf("[%s]: ExecuteSubroutineCall: error: %s", opcode, err)
	}

	// Store the result in the destination slot
	i.LogDebug("[%s]: ExecuteSubroutineCall: Storing subroutine result %v in destination %s", opcode, result, destination)
	i.Slots[destination.Name] = result
	return result, nil
}

// NewInterpreterWithParent creates a new interpreter with initial slots and opcode aliases
func (i *InterpreterImpl) NewInterpreterWithParent(
	scriptDir string,
	initialSlots map[string]interface{},
) primitive_types2.Interpreter {
	return NewInterpreterWithParent(scriptDir, initialSlots, i)
}
