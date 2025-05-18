package primitives

import (
	"fmt"
	"strings"
)

// Trim represents the trim primitive
type Trim struct{}

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
	RegisterPrimitive(&Trim{})
}
