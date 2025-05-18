package primitives

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

type StringType struct{}

var _ primitive_types.Primitive = &StringType{}

func (s *StringType) Name() string {
	return "/gnd/string"
}

func (s *StringType) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("string expects 1 argument, got %d", len(args))
	}
	arg := args[0]
	if arg == nil {
		return "", nil
	}
	return fmt.Sprintf("%v", arg), nil
}

func (s *StringType) String() string {
	return "string"
}
