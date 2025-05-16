package primitive

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/log"
)

var FirstExpectedArgsError = fmt.Errorf("first expects at least 1 argument, got 0")
var FirstExpectedItemsError = fmt.Errorf("first expects at least 1 item in an array, got 0")

// First is a primitive which takes the first element out of a list argument
type First struct{}

func (t *First) Name() string {
	return "/gnd/first"
}

func (t *First) Execute(args []interface{}) (interface{}, error) {
	log.Printf(log.Debug, "[/gnd/first]: Execute: input args=%v (type: %T)", args, args)

	if len(args) == 0 {
		return nil, FirstExpectedArgsError
	}

	a := args[0]
	if arr, ok := a.([]interface{}); ok {
		if len(arr) == 0 {
			return nil, FirstExpectedItemsError
		}
		f := arr[0]
		log.Printf(log.Debug, "[/gnd/first]: Execute: final result=%v (type: %T)", f, f)
		return f, nil
	}

	log.Printf(log.Debug, "[/gnd/first]: Execute: final result=%v (type: %T)", a, a)
	return a, nil
}

func init() {
	RegisterPrimitive(&First{})
}
