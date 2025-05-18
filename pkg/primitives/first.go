package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var FirstExpectedArgsError = fmt.Errorf("first expects at least 1 argument, got 0")
var FirstExpectedItemsError = fmt.Errorf("first expects at least 1 item in an array, got 0")

// First is a primitive which takes the first element out of a list argument
type First struct{}

var _ primitive_types.Primitive = &First{}

func (t *First) Name() string {
	return "/gnd/first"
}

func (t *First) Execute(args []interface{}) (interface{}, error) {
	loggers.Printf(loggers.Debug, "[/gnd/first]: Execute: input args=%v (type: %T)", args, args)

	if len(args) == 0 {
		return nil, FirstExpectedArgsError
	}

	a := args[0]
	if arr, ok := a.([]interface{}); ok {
		if len(arr) == 0 {
			return nil, FirstExpectedItemsError
		}
		f := arr[0]
		loggers.Printf(loggers.Debug, "[/gnd/first]: Execute: final result=%v (type: %T)", f, f)
		return f, nil
	}

	loggers.Printf(loggers.Debug, "[/gnd/first]: Execute: final result=%v (type: %T)", a, a)
	return a, nil
}

func init() {
	primitive_services.RegisterPrimitive(&First{})
}
