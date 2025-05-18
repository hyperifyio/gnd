package primitives

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

type Int16Type struct{}

var _ primitive_types.Primitive = &Int16Type{}

func (i *Int16Type) Name() string {
	return "/gnd/int16"
}

func (i *Int16Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("int16 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct int16
	if value, ok := arg.(int16); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < math.MinInt16 || value > math.MaxInt16 {
			return nil, fmt.Errorf("int16 overflow: value %d outside range -32768..32767", value)
		}
		return int16(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("int16 does not allow fractional values: %v", value)
		}
		if value < math.MinInt16 || value > math.MaxInt16 {
			return nil, fmt.Errorf("int16 overflow: value %v outside range -32768..32767", value)
		}
		return int16(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("int16 does not allow fractional values: %v", value)
		}
		if value < math.MinInt16 || value > math.MaxInt16 {
			return nil, fmt.Errorf("int16 overflow: value %v outside range -32768..32767", value)
		}
		return int16(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseInt(str, 0, 16)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for int16: %v", str)
			}
			return int16(value), nil
		}

		// Handle decimal string
		value, err := strconv.ParseInt(str, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("invalid string for int16: %v", str)
		}
		return int16(value), nil
	}

	return nil, fmt.Errorf("int16 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (i *Int16Type) String() string {
	return "int16"
}
