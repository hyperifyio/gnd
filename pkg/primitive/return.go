package primitive

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/log"
)

// Return represents the return primitive
type Return struct {
	// IsSubroutine indicates whether this primitive is being executed in a subroutine context
	IsSubroutine bool
}

// Name returns the name of the primitive
func (r *Return) Name() string {
	return "/gnd/return"
}

// Execute implements the return operation according to return-syntax.md:
// - If no arguments provided, uses current value of _ and returns it to caller's _
// - If arguments provided, requires both destination and value
// - In main script context, returns a special result that main.go will interpret as exit signal
// - In subroutine context, returns control to caller
func (r *Return) Execute(args []interface{}) (interface{}, error) {
	log.Printf(log.Debug, "Return.Execute: args=%v, IsSubroutine=%v", args, r.IsSubroutine)

	// If no arguments provided, use current value of _ as both input and output
	if len(args) == 0 {
		// The first argument should be the current value of _
		if len(args) == 0 {
			log.Printf(log.Debug, "Return.Execute: no current value provided")
			return nil, fmt.Errorf("no current value provided")
		}
		result := map[string]interface{}{
			"value":       args[0],
			"destination": "_",
		}
		log.Printf(log.Debug, "Return.Execute: returning result=%v", result)
		return result, nil
	}

	// Get the current value of _
	currentValue, ok := args[0].(interface{})
	if !ok {
		log.Printf(log.Debug, "Return.Execute: invalid current value type %T", args[0])
		return nil, fmt.Errorf("invalid current value")
	}
	log.Printf(log.Debug, "Return.Execute: currentValue=%v", currentValue)

	// Prepare the return value
	var result map[string]interface{}

	// If only one argument provided, return current value to _
	if len(args) == 1 {
		result = map[string]interface{}{
			"value":       currentValue,
			"destination": "_",
		}
	} else {
		// If a destination is provided, it must be a string
		dest, ok := args[1].(string)
		if !ok {
			log.Printf(log.Debug, "Return.Execute: invalid destination type %T", args[1])
			return nil, fmt.Errorf("destination must be a string")
		}

		// If a value is provided, use it instead of the current value
		value := currentValue
		if len(args) > 2 {
			value = args[2]
		}

		result = map[string]interface{}{
			"value":       value,
			"destination": dest,
		}
	}

	// If we're in the main script context, add a special flag to signal exit
	if !r.IsSubroutine {
		result["exit"] = true
	}

	log.Printf(log.Debug, "Return.Execute: final result=%v", result)
	return result, nil
}
