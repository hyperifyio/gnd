package primitives

import (
	"errors"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var (
	Int64NoArgumentsError     = errors.New("int64: requires exactly one argument")
	Int64InvalidArgumentError = errors.New("int64: argument must be a number or numeric string")
	Int64FractionalError      = errors.New("int64: fractional part not allowed")
	Int64OverflowError        = errors.New("int64: value out of int64 range")
)

type Int64Type struct {
}

var _ primitive_types.Primitive = &Int64Type{}

func (i *Int64Type) Name() string {
	return "/gnd/int64"
}

func (i *Int64Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, Int64NoArgumentsError
	}
	arg := args[0]
	switch v := arg.(type) {
	case int64:
		return v, nil
	case int:
		return int64(v), nil
	case float32:
		if float32(int64(v)) != v {
			return nil, Int64FractionalError
		}
		return int64(v), nil
	case float64:
		if float64(int64(v)) != v {
			return nil, Int64FractionalError
		}
		return int64(v), nil
	case string:
		str := strings.TrimSpace(v)
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			parsed, err := strconv.ParseInt(str[2:], 16, 64)
			if err != nil {
				return nil, Int64InvalidArgumentError
			}
			return parsed, nil
		}
		parsed, err := strconv.ParseInt(str, 10, 64)
		if err != nil {
			return nil, Int64InvalidArgumentError
		}
		return parsed, nil
	default:
		return nil, Int64InvalidArgumentError
	}
}

func (i *Int64Type) String() string {
	return "int64"
}
