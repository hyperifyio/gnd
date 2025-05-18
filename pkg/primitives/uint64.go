package primitives

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var (
	Uint64NoArgumentsError     = fmt.Errorf("uint64: requires exactly one argument")
	Uint64InvalidArgumentError = fmt.Errorf("uint64: argument must be a positive number")
)

type Uint64Type struct{}

var _ primitive_types.Primitive = &Uint64Type{}

func (u *Uint64Type) Name() string {
	return "/gnd/uint64"
}

func (u *Uint64Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("uint64 expects 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle direct uint64
	if value, ok := arg.(uint64); ok {
		return value, nil
	}

	// Handle int
	if value, ok := arg.(int); ok {
		if value < 0 {
			return nil, fmt.Errorf("uint64 overflow: value %d outside range 0..18446744073709551615", value)
		}
		return uint64(value), nil
	}

	// Handle float64
	if value, ok := arg.(float64); ok {
		if value != float64(int64(value)) {
			return nil, fmt.Errorf("uint64 does not allow fractional values: %v", value)
		}
		if value < 0 || value > float64(math.MaxUint64) {
			return nil, fmt.Errorf("uint64 overflow: value %v outside range 0..18446744073709551615", value)
		}
		return uint64(value), nil
	}

	// Handle float32
	if value, ok := arg.(float32); ok {
		if value != float32(int32(value)) {
			return nil, fmt.Errorf("uint64 does not allow fractional values: %v", value)
		}
		if value < 0 || value > float32(math.MaxUint64) {
			return nil, fmt.Errorf("uint64 overflow: value %v outside range 0..18446744073709551615", value)
		}
		return uint64(value), nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		// Handle hex string
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			value, err := strconv.ParseUint(str, 0, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid hex string for uint64: %v", str)
			}
			return value, nil
		}

		// Handle decimal string
		value, err := strconv.ParseUint(str, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid string for uint64: %v", str)
		}
		return value, nil
	}

	return nil, fmt.Errorf("uint64 argument must be an integer, string, or float with no fraction, got %T", arg)
}

func (u *Uint64Type) String() string {
	return "uint64"
}

type Uint64Result struct {
	Value uint64
}

func (r *Uint64Result) String() string {
	return fmt.Sprintf("%d", r.Value)
}

func (r *Uint64Result) Type() string {
	return "uint64"
}
