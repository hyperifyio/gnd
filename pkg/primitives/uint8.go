package primitives

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&Uint8Type{})
}

var (
	Uint8NoArgumentsError     = errors.New("uint8: requires exactly one argument")
	Uint8InvalidArgumentError = errors.New("uint8: argument must be a positive number between 0 and 255")
)

type Uint8Type struct{}

var _ primitive_types.Primitive = &Uint8Type{}

func (u *Uint8Type) Name() string {
	return "/gnd/uint8"
}

func (u *Uint8Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("uint8 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct uint8
	if value, ok := arg.(uint8); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < 0 || value > math.MaxUint8 {
			return nil, fmt.Errorf("uint8 overflow: value %d outside range 0..255", value)
		}
		return uint8(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("uint8 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint8 {
			return nil, fmt.Errorf("uint8 overflow: value %v outside range 0..255", value)
		}
		return uint8(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("uint8 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint8 {
			return nil, fmt.Errorf("uint8 overflow: value %v outside range 0..255", value)
		}
		return uint8(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseUint(str, 0, 8)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for uint8: %v", str)
			}
			return uint8(value), nil
		}

		// Handle decimal string
		value, err := strconv.ParseUint(str, 10, 8)
		if err != nil {
			return nil, fmt.Errorf("invalid string for uint8: %v", str)
		}
		return uint8(value), nil
	}

	return nil, fmt.Errorf("uint8 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (u *Uint8Type) String() string {
	return "uint8"
}
