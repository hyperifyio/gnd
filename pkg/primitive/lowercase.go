package primitive

import (
	"fmt"
	"strings"
)

// Lowercase represents the lowercase primitive
type Lowercase struct{}

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
	RegisterPrimitive(&Lowercase{})
}
