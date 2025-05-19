package primitives

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
)

// Select represents the select primitive
type Select struct{}

func (s *Select) Name() string {
	return "/gnd/select"
}

func (s *Select) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("select expects exactly 3 arguments (condition, trueValue, falseValue), got %d", len(args))
	}

	condition := fmt.Sprintf("%v", args[0])
	trueValue := args[1]
	falseValue := args[2]

	// Case-sensitive comparison with "true"
	if condition == "true" {
		return trueValue, nil
	}
	return falseValue, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Select{})
}
