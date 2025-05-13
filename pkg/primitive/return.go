package primitive

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/log"
)

// Return represents the return primitive
type Return struct {
}

// Name returns the name of the primitive
func (r *Return) Name() string {
	return "/gnd/return"
}

// Execute executes the return primitive
func (r *Return) Execute(args []interface{}) (interface{}, error) {
	log.Printf(log.Debug, "Return.Execute: args=%v", args)

	l := len(args)

	// If no arguments provided, it's an error
	if l == 0 {
		return nil, fmt.Errorf("return: no arguments provided")
	}

	if l != 1 {
		log.Printf(log.Warn, "return: ignoring extra arguments, only the first argument will be returned")
	}

	currentValue := args[0]
	log.Printf(log.Debug, "Return.Execute: final result=%v (type: %T)", currentValue, currentValue)
	return nil, NewReturnValue(currentValue)
}

func init() {
	RegisterPrimitive(&Return{})
}
