package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"strings"
)

// Uppercase represents the uppercase primitive
type Uppercase struct{}

var _ primitive_types.Primitive = &Uppercase{}

func (u *Uppercase) Name() string {
	return "/gnd/uppercase"
}

func (u *Uppercase) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("uppercase expects at least 1 argument, got 0")
	}

	// Convert the input to string and then to uppercase
	input := fmt.Sprintf("%v", args[0])
	return strings.ToUpper(input), nil
}

func init() {
	primitive_services.RegisterPrimitive(&Uppercase{})
}
