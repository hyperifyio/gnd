package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Exit represents the exit primitive
type Exit struct {
}

var _ primitive_types.Primitive = &Exit{}
var _ primitive_types.BlockErrorResultHandler = &Exit{}

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

// HandleBlockErrorResult handles exit errors
func (e *Exit) HandleBlockErrorResult(err error, i primitive_types.Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error) {
	if exitResult, ok := GetExitResult(err); ok {
		i.LogDebug("[/gnd/exit]: exit result detected with code %d", exitResult.Code)
		return nil, exitResult
	}
	return nil, err
}

func init() {
	primitive_services.RegisterPrimitive(&Exit{})
}
