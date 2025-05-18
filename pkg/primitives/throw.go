package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Predefined errors
var ThrowErrNoArguments = errors.New("throw: requires at least one argument")
var ThrowInvalidArgument = errors.New("throw: invalid argument")

// Throw represents the throw primitive
type Throw struct{}

var _ primitive_types.Primitive = &Throw{}

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
	str := ""
	for i, arg := range args {
		if i != 0 {
			str += " "
		}
		s, err := parsers.ParseString(arg)
		if err != nil {
			loggers.Printf(loggers.Error, "throw: invalid argument: %v", err)
			return nil, ThrowInvalidArgument
		}
		str += s
	}

	// Return an error with the composed message
	return nil, errors.New(str)
}

func init() {
	primitive_services.RegisterPrimitive(&Throw{})
}
