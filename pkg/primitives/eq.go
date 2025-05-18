package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"reflect"
)

var EqRequiresAtLeastTwoArguments = fmt.Errorf("eq expects at least 2 arguments, got 0")

// Eq represents the eq primitive
type Eq struct{}

var _ primitive_types.Primitive = &Eq{}

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
	primitive_services.RegisterPrimitive(&Eq{})
}
