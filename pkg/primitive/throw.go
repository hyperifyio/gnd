package primitive

import (
	"fmt"
	"strings"
)

// Throw represents the throw primitive
type Throw struct{}

// Name returns the name of the primitive
func (t *Throw) Name() string {
	return "/gnd/throw"
}

// Execute runs the throw primitive
func (t *Throw) Execute(args []interface{}) (interface{}, error) {
	// If no arguments provided, use the current value of _
	if len(args) == 0 {
		return nil, fmt.Errorf("_")
	}

	// Convert all arguments to strings and join them with spaces
	parts := make([]string, len(args))
	for i, arg := range args {
		parts[i] = fmt.Sprintf("%v", arg)
	}
	message := strings.Join(parts, " ")

	// Return an error with the composed message
	return nil, fmt.Errorf(message)
}

func init() {
	RegisterPrimitive(&Throw{})
}
