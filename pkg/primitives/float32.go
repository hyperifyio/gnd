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
	primitive_services.RegisterPrimitive(&Float32Type{})
}

var (
	Float32NoArgumentsError     = errors.New("float32: requires exactly one argument")
	Float32InvalidArgumentError = errors.New("float32: argument must be a number")
)

type Float32Type struct{}

var _ primitive_types.Primitive = &Float32Type{}

func (f *Float32Type) Name() string {
	return "/gnd/float32"
}

func (f *Float32Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("float32 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct float32
	if value, ok := arg.(float32); ok {
		if math.IsInf(float64(value), 0) || math.IsNaN(float64(value)) {
			return nil, fmt.Errorf("float32 value must be finite, got %v", value)
		}
		return value, nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if math.IsInf(value, 0) || math.IsNaN(value) {
			return nil, fmt.Errorf("float32 value must be finite, got %v", value)
		}
		if value > math.MaxFloat32 || value < -math.MaxFloat32 {
			return nil, fmt.Errorf("float32 overflow: value %v outside range ±3.40282346638528859811704183484516925440e+38", value)
		}
		return float32(value), nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if float64(value) > math.MaxFloat32 || float64(value) < -math.MaxFloat32 {
			return nil, fmt.Errorf("float32 overflow: value %v outside range ±3.40282346638528859811704183484516925440e+38", value)
		}
		return float32(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		value, err := strconv.ParseFloat(str, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid string for float32: %v", str)
		}
		if math.IsInf(value, 0) || math.IsNaN(value) {
			return nil, fmt.Errorf("float32 value must be finite, got %v", value)
		}
		if value > math.MaxFloat32 || value < -math.MaxFloat32 {
			return nil, fmt.Errorf("float32 overflow: value %v outside range ±3.40282346638528859811704183484516925440e+38", value)
		}
		return float32(value), nil
	}

	return nil, fmt.Errorf("float32 argument must be a number or string, got %T", arg)
}

func (f *Float32Type) String() string {
	return "float32"
}
