package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"strings"
)

// Lowercase represents the lowercase primitive
type Lowercase struct{}

var _ primitive_types.Primitive = &Lowercase{}

func (l *Lowercase) Name() string {
	return "/gnd/lowercase"
}

func (l *Lowercase) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("lowercase expects at least 1 argument, got 0")
	}

	// Convert the input to string and then to lowercase
	input := fmt.Sprintf("%v", args[0])
	return strings.ToLower(input), nil
}

func init() {
	primitive_services.RegisterPrimitive(&Lowercase{})
}
