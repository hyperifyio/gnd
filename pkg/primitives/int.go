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
	primitive_services.RegisterPrimitive(&IntType{})
}

// IntType represents the int primitive type
type IntType struct{}

var _ primitive_types.Primitive = &IntType{}

// Name returns the type name
func (i *IntType) Name() string {
	return "/gnd/int"
}

// Execute converts the input to an int
func (i *IntType) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("int expects 1 argument, got %d", len(args))
	}

	// Handle nil input
	if args[0] == nil {
		return nil, fmt.Errorf("int argument must be an integer, got <nil>")
	}

	// Try to convert the input to int
	var value int
	var err error

	switch v := args[0].(type) {
	case int:
		value = v
	case int64:
		value = int(v)
	case float32:
		// Check if float has fractional part
		if v != float32(int32(v)) {
			return nil, fmt.Errorf("int argument must be an integer, got float with fractional part")
		}
		value = int(v)
	case float64:
		// Check if float has fractional part
		if v != math.Trunc(v) {
			return nil, fmt.Errorf("int argument must be an integer, got float with fractional part")
		}
		value = int(v)
	case string:
		// Handle hex and decimal strings
		if strings.HasPrefix(v, "0x") || strings.HasPrefix(v, "-0x") {
			var parsed int64
			parsed, err = strconv.ParseInt(v, 0, 64)
			value = int(parsed)
		} else {
			value, err = strconv.Atoi(v)
		}
		if err != nil {
			return nil, fmt.Errorf("int argument must be an integer, got invalid string: %v", err)
		}
	default:
		return nil, fmt.Errorf("int argument must be an integer, got %T", args[0])
	}

	// Only enforce 32-bit range on 32-bit builds
	if IntSize() == 32 {
		if value > math.MaxInt32 || value < math.MinInt32 {
			return nil, fmt.Errorf("overflow outside 32-bit range")
		}
	}

	return value, nil
}

// IntSize returns the size of int in bits for the current platform
func IntSize() int {
	if strconv.IntSize == 32 {
		return 32
	}
	return 64
}

func (i *IntType) String() string {
	return "int"
}
