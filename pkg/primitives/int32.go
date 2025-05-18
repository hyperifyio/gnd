package primitives

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

type Int32Type struct{}

var _ primitive_types.Primitive = &Int32Type{}

func (i *Int32Type) Name() string {
	return "/gnd/int32"
}

func (i *Int32Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("int32 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct int32
	if value, ok := arg.(int32); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < math.MinInt32 || value > math.MaxInt32 {
			return nil, fmt.Errorf("int32 overflow: value %d outside range -2147483648..2147483647", value)
		}
		return int32(value), nil
	}

	// Handle int64
	if value, ok := arg.(int64); ok {
		if value < math.MinInt32 || value > math.MaxInt32 {
			return nil, fmt.Errorf("int32 overflow: value %d outside range -2147483648..2147483647", value)
		}
		return int32(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("int32 does not allow fractional values: %v", value)
		}
		if value < math.MinInt32 || value > math.MaxInt32 {
			return nil, fmt.Errorf("int32 overflow: value %v outside range -2147483648..2147483647", value)
		}
		return int32(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("int32 does not allow fractional values: %v", value)
		}
		if value < math.MinInt32 || value > math.MaxInt32 {
			return nil, fmt.Errorf("int32 overflow: value %v outside range -2147483648..2147483647", value)
		}
		return int32(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		str = strings.TrimSpace(str)
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseInt(str, 0, 32)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for int32: %v", str)
			}
			return int32(value), nil
		}

		// Handle decimal string
		value, err := strconv.ParseInt(str, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid string for int32: %v", str)
		}
		return int32(value), nil
	}

	return nil, fmt.Errorf("int32 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (i *Int32Type) String() string {
	return "int32"
}
