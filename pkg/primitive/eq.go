package primitive

import (
	"fmt"
	"reflect"
)

var EqRequiresAtLeastTwoArguments = fmt.Errorf("eq expects at least 2 arguments, got 0")

// Eq represents the eq primitive
type Eq struct{}

func (e *Eq) Name() string {
	return "/gnd/eq"
}

func (e *Eq) Execute(args []interface{}) (interface{}, error) {
	if len(args) < 2 {
		return nil, EqRequiresAtLeastTwoArguments
	}

	// Get the first value to compare against
	firstValue := args[0]

	// Compare all other values against the first value
	for i := 1; i < len(args); i++ {
		if !reflect.DeepEqual(firstValue, args[i]) {
			return false, nil
		}
	}

	return true, nil
}

func init() {
	RegisterPrimitive(&Eq{})
}
