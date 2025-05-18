package primitives

import (
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&StringType{})
}

type StringType struct{}

var _ primitive_types.Primitive = &StringType{}

func (s *StringType) Name() string {
	return "/gnd/string"
}

func (s *StringType) Execute(args []interface{}) (interface{}, error) {
	l := len(args)
	if l == 0 {
		return "", nil
	} else {
		str := ""
		for i, arg := range args {
			if i != 0 {
				str += "\n"
			}
			s, err := parsers.ParseString(arg)
			if err != nil {
				return nil, err
			}
			str += s
		}
		return str, nil
	}
}

func (s *StringType) String() string {
	return "string"
}
