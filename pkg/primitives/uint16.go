package primitives

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&Uint16Type{})
}

type Uint16Type struct{}

var _ primitive_types.Primitive = &Uint16Type{}

func (u *Uint16Type) Name() string {
	return "/gnd/uint16"
}

func (u *Uint16Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("uint16 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct uint16
	if value, ok := arg.(uint16); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < 0 || value > math.MaxUint16 {
			return nil, fmt.Errorf("uint16 overflow: value %d outside range 0..65535", value)
		}
		return uint16(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("uint16 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint16 {
			return nil, fmt.Errorf("uint16 overflow: value %v outside range 0..65535", value)
		}
		return uint16(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("uint16 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint16 {
			return nil, fmt.Errorf("uint16 overflow: value %v outside range 0..65535", value)
		}
		return uint16(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseUint(str, 0, 16)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for uint16: %v", str)
			}
			return uint16(value), nil
		}

		// Handle decimal string
		value, err := strconv.ParseUint(str, 10, 16)
		if err != nil {
			return nil, fmt.Errorf("invalid string for uint16: %v", str)
		}
		return uint16(value), nil
	}

	return nil, fmt.Errorf("uint16 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (u *Uint16Type) String() string {
	return "uint16"
}
