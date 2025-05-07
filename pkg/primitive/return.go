package primitive

import (
	"fmt"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
)

// Return represents the return primitive
type Return struct {
	// IsSubroutine indicates whether this primitive is being executed in a subroutine context
	IsSubroutine bool
	// Destination is the destination for the return operation
	Destination string
}

// Name returns the name of the primitive
func (r *Return) Name() string {
	return "/gnd/return"
}

// Execute executes the return primitive
func (r *Return) Execute(args []interface{}) (interface{}, error) {
	log.Printf(log.Debug, "Return.Execute: args=%v, IsSubroutine=%v", args, r.IsSubroutine)

	// If no arguments provided, use current value of _ as both input and output
	if len(args) == 0 {
		log.Printf(log.Debug, "Return.Execute: no arguments provided")
		return nil, fmt.Errorf("no current value provided")
	}

	// Get the current value of _
	currentValue := args[0]
	if currentValue == nil {
		log.Printf(log.Debug, "Return.Execute: invalid current value (nil)")
		return nil, fmt.Errorf("invalid current value")
	}
	log.Printf(log.Debug, "Return.Execute: currentValue=%v (type: %T)", currentValue, currentValue)

	// If only one argument provided, return current value to _
	if len(args) == 1 {
		log.Printf(log.Debug, "Return.Execute: single argument, returning currentValue=%v (type: %T)", currentValue, currentValue)
		return map[string]interface{}{
			"value":       currentValue,
			"destination": r.Destination,
		}, nil
	}

	// If multiple values are provided, join them with spaces
	var value interface{}
	if len(args) > 1 {
		// Convert all arguments to strings and join them
		strArgs := make([]string, len(args))
		for i, arg := range args {
			if str, ok := arg.(string); ok {
				strArgs[i] = str
			} else {
				strArgs[i] = fmt.Sprintf("%v", arg)
			}
		}
		value = strings.Join(strArgs, " ")
		log.Printf(log.Debug, "Return.Execute: multiple values, joining with spaces: %v (type: %T)", value, value)
	} else {
		value = currentValue
		log.Printf(log.Debug, "Return.Execute: single value: %v (type: %T)", value, value)
	}

	result := map[string]interface{}{
		"value":       value,
		"destination": r.Destination,
	}

	// If we're in the main script context, add a special flag to signal exit
	if !r.IsSubroutine {
		result["exit"] = true
	}

	log.Printf(log.Debug, "Return.Execute: final result=%v (type: %T)", result, result)
	return result, nil
}

// NewReturn creates a new return primitive
func NewReturn(isSubroutine bool, destination string) *Return {
	return &Return{
		IsSubroutine: isSubroutine,
		Destination:  destination,
	}
}

func init() {
	Register(&Return{})
}
