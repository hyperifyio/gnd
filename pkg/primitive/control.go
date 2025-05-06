package primitive

import (
	"encoding/json"
	"fmt"
)

type controlPrimitive struct{}

func (p *controlPrimitive) Name() string {
	return "/gnd/control"
}

func (p *controlPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("control expects 3 arguments")
	}
	return ControlValue(args[0], args[1], args[2])
}

// ControlValue executes a function based on a condition
func ControlValue(cond, then, elseFn string) (interface{}, error) {
	// Parse the condition
	var b bool
	if err := json.Unmarshal([]byte(cond), &b); err != nil {
		return nil, fmt.Errorf("invalid condition: %v", err)
	}

	// Execute the appropriate branch
	if b {
		// Parse the then branch
		var thenFn string
		if err := json.Unmarshal([]byte(then), &thenFn); err != nil {
			return nil, fmt.Errorf("invalid then branch: %v", err)
		}

		// Call the then function
		prim, ok := Get(thenFn)
		if !ok {
			return nil, fmt.Errorf("unknown function: %s", thenFn)
		}

		// Execute the function
		return prim.Execute([]string{})
	}

	// Parse the else branch
	var elseFnStr string
	if err := json.Unmarshal([]byte(elseFn), &elseFnStr); err != nil {
		return nil, fmt.Errorf("invalid else branch: %v", err)
	}

	// Call the else function
	prim, ok := Get(elseFnStr)
	if !ok {
		return nil, fmt.Errorf("unknown function: %s", elseFnStr)
	}

	// Execute the function
	return prim.Execute([]string{})
}

func init() {
	RegisterPrimitive(&controlPrimitive{})
}
