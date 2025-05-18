package primitives

import (
	"fmt"
	"os"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Print represents the print primitive
type Print struct{}

var _ primitive_types.Primitive = &Print{}

// Name returns the name of the primitive
func (p *Print) Name() string {
	return "/gnd/print"
}

// Execute runs the print primitive
func (p *Print) Execute(args []interface{}) (interface{}, error) {
	str := ""
	l := len(args)
	if l != 0 {
		for i, arg := range args {
			if i != 0 {
				str += " "
			}
			s, err := parsers.ParseString(arg)
			if err != nil {
				return nil, fmt.Errorf("print: %v", err)
			}
			str += s
		}
	}
	fmt.Fprint(os.Stdout, str)
	return str, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Print{})
}
