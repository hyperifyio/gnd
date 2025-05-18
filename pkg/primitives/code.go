package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/helpers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	primitive_types2 "github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Code represents the code primitive
type Code struct {
}

var _ primitive_types2.BlockSuccessResultHandler = &Code{}

// Name returns the name of the primitive
func (c *Code) Name() string {
	return "/gnd/code"
}

// Execute runs the code primitive
func (c *Code) Execute(args []interface{}) (interface{}, error) {

	// If no arguments, return current routine's instructions
	if len(args) == 0 {
		return NewCodeResult([]interface{}{"@"}), nil
	}

	// Process all arguments
	targets := make([]interface{}, len(args))
	for i, arg := range args {
		switch v := arg.(type) {
		case string:
			targets[i] = v
		case []*parsers.Instruction:
			targets[i] = v
		default:
			return nil, fmt.Errorf("code: target must be a string or instruction array, got %T", arg)
		}
	}

	// Return a request to resolve all targets
	return NewCodeResult(targets), nil
}

// HandleBlockSuccessResult handles code results
func (c *Code) HandleBlockSuccessResult(result interface{}, i primitive_types2.Interpreter, destination *parsers.PropertyRef, instructions []*parsers.Instruction) (interface{}, error) {
	if codeResult, ok := GetCodeResult(result); ok {

		codeInstructions, err := HandleCodeResult(i, "/gnd/code", codeResult, instructions)
		if err != nil {
			return nil, err
		}

		// Store the codeInstructions in the destination slot
		i.LogDebug("[/gnd/code]: ExecutePrimitive: storing %d codeInstructions in %s", len(codeInstructions), destination)
		i.SetSlot(destination.Name, codeInstructions)

		return codeInstructions, nil
	}
	return result, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Code{})
}

// HandleCodeResult processes a CodeResult and returns the concatenated instructions
func HandleCodeResult(i primitive_types2.Interpreter, source string, codeResult *CodeResult, block []*parsers.Instruction) ([]*parsers.Instruction, error) {

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
				subPath := helpers.SubroutinePath(v, pwd)
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
