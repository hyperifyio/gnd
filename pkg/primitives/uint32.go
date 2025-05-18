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
	primitive_services.RegisterPrimitive(&Uint32Type{})
}

var (
	Uint32NoArgumentsError     = errors.New("uint32: requires exactly one argument")
	Uint32InvalidArgumentError = errors.New("uint32: argument must be a positive number between 0 and 4294967295")
)

type Uint32Type struct{}

var _ primitive_types.Primitive = &Uint32Type{}

func (u *Uint32Type) Name() string {
	return "/gnd/uint32"
}

func (u *Uint32Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("uint32 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct uint32
	if value, ok := arg.(uint32); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < 0 || uint64(value) > math.MaxUint32 {
			return nil, fmt.Errorf("uint32 overflow: value %d outside range 0..4294967295", value)
		}
		return uint32(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("uint32 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint32 {
			return nil, fmt.Errorf("uint32 overflow: value %v outside range 0..4294967295", value)
		}
		return uint32(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("uint32 does not allow fractional values: %v", value)
		}
		if value < 0 || value > math.MaxUint32 {
			return nil, fmt.Errorf("uint32 overflow: value %v outside range 0..4294967295", value)
		}
		return uint32(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseUint(str, 0, 32)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for uint32: %v", str)
			}
			return uint32(value), nil
		}

		// Handle decimal string
		value, err := strconv.ParseUint(str, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("invalid string for uint32: %v", str)
		}
		return uint32(value), nil
	}

	return nil, fmt.Errorf("uint32 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (u *Uint32Type) String() string {
	return "uint32"
}
