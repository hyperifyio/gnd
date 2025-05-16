package primitive

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/parsers"
)

// Exit represents the exit primitive
type Exit struct{}

// Name returns the name of the primitive
func (e *Exit) Name() string {
	return "/gnd/exit"
}

// Execute runs the exit primitive
func (e *Exit) Execute(args []interface{}) (interface{}, error) {

	// If no arguments provided, exit with code 1
	if len(args) == 0 {
		return nil, NewExitResult(1, nil)
	}

	// If one argument provided, it must be an integer
	value, err := parsers.ParseInt(args[0])
	if err != nil {
		return nil, fmt.Errorf("[/gnd/exit]: exit code invalid: %v", err)
	}
	return nil, NewExitResult(value, nil)
}

func init() {
	RegisterPrimitive(&Exit{})
}
