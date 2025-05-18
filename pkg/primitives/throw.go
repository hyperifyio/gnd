package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
)

// Predefined errors
var ThrowErrNoArguments = errors.New("throw: requires at least one argument")
var ThrowInvalidArgument = errors.New("throw: invalid argument")

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
		loggers.Printf(loggers.Error, "throw: invalid argument: %v", err)
		return nil, ThrowInvalidArgument
	}

	// Return an error with the composed message
	return nil, errors.New(message)
}

func init() {
	primitive_services.RegisterPrimitive(&Throw{})
}
