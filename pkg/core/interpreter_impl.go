package core

import (
	"fmt"
	"io/fs"
	"os"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive"
	"github.com/hyperifyio/gnd/pkg/units"
)

// InterpreterImpl represents the execution environment
type InterpreterImpl struct {
	Slots       map[string]interface{}
	Subroutines map[string][]*parsers.Instruction
	ScriptDir   string            // Directory of the currently executing script
	LogIndent   int               // Current log indentation level
	UnitsFS     fs.FS             // Embedded filesystem containing GND units
	OpcodeMap   map[string]string // Map of opcode aliases
	parent      *InterpreterImpl  // Parent interpreter for nested calls
}

// NewInterpreter creates a new core instance
func NewInterpreter(
	scriptDir string,
	opcodeMap map[string]string,
) Interpreter {
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
	parent *InterpreterImpl,
) Interpreter {
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

			result, err := i.ExecuteInstruction(op.Opcode, op.Destination, op.Arguments)

			if err != nil {
				if returnValue, ok := primitive.GetReturnValue(err); ok {
					i.LogDebug("[%s:%d]: ExecuteInstructionBlock: return by ReturnValue: %v", source, idx, returnValue.Value)
					return returnValue.Value, nil
				}
				return nil, fmt.Errorf("\n  %s:%d: %v", source, idx, err)
			}

			if codeResult, ok := primitive.GetCodeResult(result); ok {
				codeInstructions, err := i.HandleCodeResult(source, codeResult, instructions)
				if err != nil {
					return nil, err
				}

				// Store the codeInstructions in the destination slot
				i.LogDebug("[%s]: ExecutePrimitive: storing %d codeInstructions in %s", source, len(codeInstructions), op.Destination.Name)
				i.Slots[op.Destination.Name] = codeInstructions
				return codeInstructions, nil
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
	subPath := SubroutinePath(name, pwd)
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

// ExecuteInstruction executes a single GND instruction and returns its result
func (i *InterpreterImpl) ExecuteInstruction(opcode string, destination *parsers.PropertyRef, arguments []interface{}) (interface{}, error) {

	i.LogDebug("[%s]: ExecuteInstruction: preparing: %s %v", opcode, opcode, arguments)

	// Check if the opcode exists in the default alias map
	opcode = i.ResolveOpcode(opcode)

	// Resolve arguments by mapping context properties
	resolvedArgs, err := i.LoadArguments(opcode, arguments)
	if err != nil {
		return nil, fmt.Errorf("[%s]: ExecuteInstruction: failed to load arguments: %s", opcode, err)
	}
	i.LogDebug("[%s]: ExecuteInstruction: Resolved arguments as: %v from %v", opcode, resolvedArgs, arguments)

	prim, ok := primitive.Get(opcode)
	if !ok {
		i.LogDebug("[%s]: ExecuteInstruction: subroutine: %v <- %s %v", opcode, destination, opcode, resolvedArgs)
		return i.ExecuteSubroutineCall(opcode, destination, resolvedArgs)
	}

	i.LogDebug("[%s]: ExecuteInstruction: primitive: %v <- %s %v", opcode, opcode, destination, resolvedArgs)
	return i.ExecutePrimitive(prim, destination, resolvedArgs)
}

func (i *InterpreterImpl) LoadArguments(source string, arguments []interface{}) ([]interface{}, error) {

	// Resolve arguments by mapping context properties
	resolvedArgs, err := parsers.MapContextProperties(source, i.Slots, arguments)
	if err != nil {
		return nil, err
	}

	i.LogDebug("[%s]: LoadArguments: %v from %v", source, resolvedArgs, arguments)
	return resolvedArgs, nil
}

// HandleCodeResult processes a CodeResult and returns the concatenated instructions
func (i *InterpreterImpl) HandleCodeResult(source string, codeResult *primitive.CodeResult, block []*parsers.Instruction) ([]*parsers.Instruction, error) {
	i.LogDebug("[%s]: HandleCodeResult: processing targets: %v", source, codeResult.Targets)
	var allInstructions []*parsers.Instruction

	// Process each target in order
	for _, target := range codeResult.Targets {
		var instructions []*parsers.Instruction

		switch v := target.(type) {
		case string:
			if v == "@" {
				if block == nil {
					return nil, fmt.Errorf("[%s]: HandleCodeResult: no instructions provided for @", source)
				}
				instructions = append(instructions, block...)
			} else {

				// Get instructions using existing subroutine logic
				pwd := i.GetScriptDir()
				i.LogDebug("[%s]: HandleCodeResult: pwd = %s", source, pwd)

				// Check if the subroutine is already loaded
				subPath := SubroutinePath(v, pwd)
				i.LogDebug("[%s]: HandleCodeResult: subPath = %v", source, subPath)

				var err error
				instructions, err = i.GetSubroutineInstructions(subPath)
				i.LogDebug("[%s]: HandleCodeResult: instructions = %v", source, instructions)
				if err != nil {
					return nil, fmt.Errorf("[%s]: HandleCodeResult: failed to get instructions for %v: %v", source, target, err)
				}
				if instructions == nil {
					return nil, fmt.Errorf("[%s]: HandleCodeResult: no instructions found for %v", source, target)
				}

			}
		case []*parsers.Instruction:
			// Use the instructions directly
			instructions = v
			if instructions == nil {
				return nil, fmt.Errorf("[%s]: HandleCodeResult: no instructions found for %v", source, target)
			}
		case *parsers.Instruction:
			// Use the instructions directly
			instructions = []*parsers.Instruction{v}
			if instructions == nil {
				return nil, fmt.Errorf("[%s]: HandleCodeResult: no instructions found for %v", source, target)
			}
		default:
			return nil, fmt.Errorf("[%s]: HandleCodeResult: invalid target type: %T", source, target)
		}

		// Append instructions to the result
		allInstructions = append(allInstructions, instructions...)
	}

	return allInstructions, nil
}

// ExecutePrimitive executes a single GND primitive and returns its result
func (i *InterpreterImpl) ExecutePrimitive(prim primitive.Primitive, destination *parsers.PropertyRef, arguments []interface{}) (interface{}, error) {

	// Log regular instruction
	i.LogDebug("[%s]: ExecutePrimitive: %v <- %s %v", prim.Name(), destination, prim.Name(), arguments)

	result, err := prim.Execute(arguments)

	if err != nil {

		// Check if this is a ReturnValue
		if returnValue, ok := primitive.GetReturnValue(err); ok {
			i.LogDebug("[%s]: ExecutePrimitive: exit result detected with value %v", prim.Name(), returnValue.Value)
			i.Slots[destination.Name] = returnValue.Value
			return nil, returnValue
		}

		// Check if this is an ExitResult
		if exitResult, ok := primitive.GetExitResult(err); ok {
			i.LogDebug("[%s]: ExecutePrimitive: exit result detected with code %d", prim.Name(), exitResult.Code)
			// Store the value in the destination slot before exiting
			if exitResult.Value != nil {
				i.LogDebug("[%s]: ExecutePrimitive: storing value %v (type: %T) in destination %s", prim.Name(), exitResult.Value, exitResult.Value, destination)
				i.Slots[destination.Name] = exitResult.Value
				i.LogDebug("[%s]: ExecutePrimitive: after storing, destination %s contains: %v (type: %T)", prim.Name(), destination, i.Slots[destination.Name], i.Slots[destination.Name])
				return nil, exitResult
			}
			return nil, exitResult
		}

		return nil, fmt.Errorf("[%s]: ExecutePrimitive: error: %v", prim.Name(), err)
	}

	// Check if this is a CodeResult
	if codeResult, ok := primitive.GetCodeResult(result); ok {
		i.LogDebug("[%s]: ExecutePrimitive: code result detected: %v", prim.Name(), codeResult)
		return codeResult, nil
	}

	// Store the result in the destination slot
	i.LogDebug("[%s]: ExecutePrimitive: %v <- %v", prim.Name(), destination, log.StringifyValue(result))
	i.Slots[destination.Name] = result
	return result, nil
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
