package primitives

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&Float64Type{})
}

var (
	Float64NoArgumentsError     = errors.New("float64: requires exactly one argument")
	Float64InvalidArgumentError = errors.New("float64: argument must be a number")
)

type Float64Type struct{}

var _ primitive_types.Primitive = &Float64Type{}

func (f *Float64Type) Name() string {
	return "/gnd/float64"
}

func (f *Float64Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("float64 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct float64
	if value, ok := arg.(float64); ok {
		if math.IsInf(value, 0) || math.IsNaN(value) {
			return nil, fmt.Errorf("float64 value must be finite, got %v", value)
		}
		return value, nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if math.IsInf(float64(value), 0) || math.IsNaN(float64(value)) {
			return nil, fmt.Errorf("float64 value must be finite, got %v", value)
		}
		return float64(value), nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		return float64(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		value, err := strconv.ParseFloat(str, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid string for float64: %v", str)
		}
		if math.IsInf(value, 0) || math.IsNaN(value) {
			return nil, fmt.Errorf("float64 value must be finite, got %v", value)
		}
		if value > math.MaxFloat64 || value < -math.MaxFloat64 {
			return nil, fmt.Errorf("float64 overflow: value %v outside range ±1.7976931348623157e+308", value)
		}
		return value, nil
	}

	return nil, fmt.Errorf("float64 argument must be a number or string, got %T", arg)
}

func (f *Float64Type) String() string {
	return "float64"
}
