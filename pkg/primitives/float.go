package primitives

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

type FloatType struct {
	Value float64
}

var _ primitive_types.Primitive = &FloatType{}

func (f *FloatType) Name() string {
	return "/gnd/float"
}

func (f *FloatType) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("float expects 1 argument, got %d", len(args))
	}
	value, ok := args[0].(float64)
	if !ok {
		return nil, fmt.Errorf("float argument must be a float64, got %T", args[0])
	}
	return &FloatResult{Value: value}, nil
}

func (f *FloatType) String() string {
	return fmt.Sprintf("float %f", f.Value)
}

type FloatResult struct {
	Value float64
}

func (r *FloatResult) String() string {
	return fmt.Sprintf("%f", r.Value)
}

func (r *FloatResult) Type() string {
	return "float"
}
