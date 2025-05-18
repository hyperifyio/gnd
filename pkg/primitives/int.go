package primitives

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

type IntType struct {
	Value int64
}

var _ primitive_types.Primitive = &IntType{}

func (i *IntType) Name() string {
	return "/gnd/int"
}

func (i *IntType) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("int expects 1 argument, got %d", len(args))
	}

	// Handle nil input
	if args[0] == nil {
		return nil, fmt.Errorf("int argument must be an int64, got <nil>")
	}

	// Try to convert the input to int64
	var value int64
	var err error

	switch v := args[0].(type) {
	case int64:
		value = v
	case string:
		// Handle hex strings
		if strings.HasPrefix(v, "0x") || strings.HasPrefix(v, "-0x") {
			value, err = strconv.ParseInt(v, 0, 64)
		} else {
			value, err = strconv.ParseInt(v, 10, 64)
		}
		if err != nil {
			return nil, fmt.Errorf("int argument must be an int64, got string")
		}
	case float64:
		// Check if float has fractional part
		if v != math.Trunc(v) {
			return nil, fmt.Errorf("int argument must be an int64, got float64")
		}
		value = int64(v)
	default:
		return nil, fmt.Errorf("int argument must be an int64, got %T", args[0])
	}

	// Only enforce 32-bit range on 32-bit builds
	if intSize() == 32 {
		if value > math.MaxInt32 || value < math.MinInt32 {
			return nil, fmt.Errorf("overflow outside 32-bit range")
		}
	}

	return &IntResult{Value: value}, nil
}

// intSize returns the size of int in bits for the current platform
func intSize() int {
	if strconv.IntSize == 32 {
		return 32
	}
	return 64
}

func (i *IntType) String() string {
	return fmt.Sprintf("int %d", i.Value)
}

type IntResult struct {
	Value int64
}

func (r *IntResult) String() string {
	return fmt.Sprintf("%d", r.Value)
}

func (r *IntResult) Type() string {
	return "int"
}
