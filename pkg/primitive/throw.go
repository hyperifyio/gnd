package primitive

import (
	"errors"
	"fmt"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// Predefined errors
var ThrowErrNoArguments = errors.New("throw: requires at least one argument")

// Throw represents the throw primitive
type Throw struct{}

// Name returns the name of the primitive
func (t *Throw) Name() string {
	return "/gnd/throw"
}

// Execute runs the throw primitive
func (t *Throw) Execute(args []interface{}) (interface{}, error) {

	// If no arguments provided, return an error
	if len(args) == 0 {
		return nil, ThrowErrNoArguments
	}

	// Convert arguments to string using ParseString
	message, err := parsers.ParseString(args)
	if err != nil {
		return nil, fmt.Errorf("invalid argument: %v", err)
	}

	// Return an error with the composed message
	return nil, errors.New(message)
}

func init() {
	RegisterPrimitive(&Throw{})
}
