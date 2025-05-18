package primitives

import (
	"errors"
	"fmt"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var CompileRequiresAtLeastOneSource = errors.New("compile: requires at least one source")

// Compile represents the compile primitive
type Compile struct{}

var _ primitive_types.Primitive = &Compile{}

// Name returns the name of the primitive
func (c *Compile) Name() string {
	return "/gnd/compile"
}

// Execute runs the compile primitive
func (c *Compile) Execute(args []interface{}) (interface{}, error) {

	// Require at least one source
	if len(args) == 0 {
		return nil, CompileRequiresAtLeastOneSource
	}

	var allInstructions []*parsers.Instruction
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			// Parse string as Gendo code
			instructions, err := parsers.ParseInstructionLines("/gnd/compile", v)
			if err != nil {
				return nil, fmt.Errorf("compile: failed to parse source: %v", err)
			}
			allInstructions = append(allInstructions, instructions...)
		case []*parsers.Instruction:
			// Use the instructions directly
			allInstructions = append(allInstructions, v...)
		case *parsers.Instruction:
			// Use the instruction directly
			allInstructions = append(allInstructions, v)
		default:
			return nil, fmt.Errorf("compile: source must be a string or instruction array, got %T", arg)
		}
	}

	return allInstructions, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Compile{})
}
