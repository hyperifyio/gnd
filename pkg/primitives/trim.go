package primitives

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"strings"
)

// Trim represents the trim primitive
type Trim struct{}

var _ primitive_types.Primitive = &Trim{}

func (t *Trim) Name() string {
	return "/gnd/trim"
}

func (t *Trim) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("trim expects at least 1 argument, got 0")
	}

	// Convert the input to string
	input := fmt.Sprintf("%v", args[0])

	// If no characters to trim specified, use default whitespace
	if len(args) == 1 {
		return strings.TrimSpace(input), nil
	}

	// Use the specified characters to trim
	chars := fmt.Sprintf("%v", args[1])
	return strings.Trim(input, chars), nil
}

func init() {
	primitive_services.RegisterPrimitive(&Trim{})
}
