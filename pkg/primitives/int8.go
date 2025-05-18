package primitives

import (
	"errors"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&Int8Type{})
}

type Int8Type struct{}

var _ primitive_types.Primitive = &Int8Type{}

var (
	Int8NoArgumentsError     = errors.New("int8: requires exactly one argument")
	Int8InvalidArgumentError = errors.New("int8: argument must be a number or numeric string")
	Int8FractionalError      = errors.New("int8: fractional part not allowed")
	Int8OverflowError        = errors.New("int8: value out of int8 range")
)

func (i *Int8Type) Name() string {
	return "/gnd/int8"
}

func (i *Int8Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, Int8NoArgumentsError
	}
	arg := args[0]
	switch v := arg.(type) {
	case int8:
		return v, nil
	case int:
		if v < -128 || v > 127 {
			return nil, Int8OverflowError
		}
		return int8(v), nil
	case float32:
		if float32(int8(v)) != v {
			return nil, Int8FractionalError
		}
		if v < -128 || v > 127 {
			return nil, Int8OverflowError
		}
		return int8(v), nil
	case float64:
		if float64(int8(v)) != v {
			return nil, Int8FractionalError
		}
		if v < -128 || v > 127 {
			return nil, Int8OverflowError
		}
		return int8(v), nil
	case string:
		str := strings.TrimSpace(v)
		var parsed int64
		var err error
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			parsed, err = strconv.ParseInt(str[2:], 16, 8)
		} else {
			parsed, err = strconv.ParseInt(str, 10, 8)
		}
		if err != nil {
			return nil, Int8InvalidArgumentError
		}
		if parsed < -128 || parsed > 127 {
			return nil, Int8OverflowError
		}
		return int8(parsed), nil
	default:
		return nil, Int8InvalidArgumentError
	}
}

func (i *Int8Type) String() string {
	return "int8"
}
