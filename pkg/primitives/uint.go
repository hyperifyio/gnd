package primitives

import (
	"errors"
	"strconv"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var (
	UintNoArgumentsError     = errors.New("uint: requires exactly one argument")
	UintInvalidArgumentError = errors.New("uint: argument must be a non-negative integer or numeric string")
	UintFractionalError      = errors.New("uint: fractional part not allowed")
	UintOverflowError        = errors.New("uint: value out of uint range")
)

type UintType struct{}

var _ primitive_types.Primitive = &UintType{}

func (u *UintType) Name() string {
	return "/gnd/uint"
}

func (u *UintType) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, UintNoArgumentsError
	}
	arg := args[0]
	switch v := arg.(type) {
	case uint:
		return v, nil
	case int:
		if v < 0 {
			return nil, UintInvalidArgumentError
		}
		return uint(v), nil
	case float32:
		if v < 0 {
			return nil, UintInvalidArgumentError
		}
		if float32(uint(v)) != v {
			return nil, UintFractionalError
		}
		return uint(v), nil
	case float64:
		if v < 0 {
			return nil, UintInvalidArgumentError
		}
		if float64(uint(v)) != v {
			return nil, UintFractionalError
		}
		return uint(v), nil
	case string:
		str := strings.TrimSpace(v)
		var parsed uint64
		var err error
		if strings.HasPrefix(str, "0x") || strings.HasPrefix(str, "0X") {
			parsed, err = strconv.ParseUint(str[2:], 16, 64)
		} else {
			parsed, err = strconv.ParseUint(str, 10, 64)
		}
		if err != nil {
			return nil, UintInvalidArgumentError
		}
		if parsed > uint64(^uint(0)) {
			return nil, UintOverflowError
		}
		return uint(parsed), nil
	default:
		return nil, UintInvalidArgumentError
	}
}

func (u *UintType) String() string {
	return "uint"
}
