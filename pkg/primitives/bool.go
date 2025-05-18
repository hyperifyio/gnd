package primitives

import (
	"fmt"
	"reflect"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&BoolType{})
}

type BoolType struct {
}

var _ primitive_types.Primitive = &BoolType{}

func (b *BoolType) Name() string {
	return "/gnd/bool"
}

func (b *BoolType) Execute(args []interface{}) (interface{}, error) {

	// Handle zero arguments
	if len(args) == 0 {
		return false, nil
	}

	if len(args) > 1 {
		return nil, fmt.Errorf("bool expects at most 1 argument, got %d", len(args))
	}

	arg := args[0]

	// Handle nil/none
	if arg == nil {
		return false, nil
	}

	// Handle boolean
	if boolVal, ok := arg.(bool); ok {
		return boolVal, nil
	}

	// Handle numbers
	switch v := arg.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return reflect.ValueOf(v).Convert(reflect.TypeOf(float64(0))).Float() != 0, nil
	}

	// Handle string
	if str, ok := arg.(string); ok {
		return len(str) > 0, nil
	}

	// Handle arrays and maps
	switch v := arg.(type) {
	case []interface{}:
		return len(v) > 0, nil
	case map[string]interface{}:
		return len(v) > 0, nil
	}

	// For any other type, return false
	return false, nil
}
