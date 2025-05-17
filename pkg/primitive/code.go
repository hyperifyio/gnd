package primitive

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// Code represents the code primitive
type Code struct{}

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

func init() {
	RegisterPrimitive(&Code{})
}
