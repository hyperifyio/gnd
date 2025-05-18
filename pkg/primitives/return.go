package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	primitive_types2 "github.com/hyperifyio/gnd/pkg/primitive_types"
)

var ReturnNoArgumentsProvidedError = errors.New("return: no arguments provided")
var ReturnIncorrectAmountOfArgumentsError = errors.New("return: too many arguments provided")

// Return represents the return primitive
type Return struct {
}

var _ primitive_types2.BlockErrorResultHandler = &Return{}

// Name returns the name of the primitive
func (r *Return) Name() string {
	return "/gnd/return"
}

// Execute executes the return primitive
func (r *Return) Execute(args []interface{}) (interface{}, error) {
	loggers.Printf(loggers.Debug, "Return.Execute: args=%v", args)

	l := len(args)

	// If no arguments provided, it's an error
	if l == 0 {
		return nil, ReturnNoArgumentsProvidedError
	}

	if l != 1 {
		return nil, ReturnIncorrectAmountOfArgumentsError
	}

	currentValue := args[0]
	loggers.Printf(loggers.Debug, "Return.Execute: final result=%v (type: %T)", currentValue, currentValue)
	return nil, NewReturnValue(currentValue)
}

// HandleBlockErrorResult handles return values
func (r *Return) HandleBlockErrorResult(err error, i primitive_types2.Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error) {
	if returnValue, ok := GetReturnValue(err); ok {
		i.LogDebug("[/gnd/return]: return value detected: %s = %v", destination, returnValue.Value)
		i.SetSlot(destination.Name, returnValue.Value)
		return returnValue, nil
	}
	return nil, err
}

func init() {
	primitive_services.RegisterPrimitive(&Return{})
}
