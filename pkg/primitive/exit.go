package primitive

import (
	"fmt"
	"strconv"
)

// Exit represents the exit primitive
type Exit struct{}

// Name returns the name of the primitive
func (e *Exit) Name() string {
	return "/gnd/exit"
}

// Execute runs the exit primitive
func (e *Exit) Execute(args []interface{}) (interface{}, error) {
	// Default exit code is 1
	exitCode := 1

	// If no arguments provided, exit with code 1
	if len(args) == 0 {
		return map[string]interface{}{
			"exit": true,
			"code": exitCode,
		}, nil
	}

	// If one argument provided, it must be an integer
	if len(args) == 1 {
		// Try to convert the argument to an integer
		switch v := args[0].(type) {
		case int:
			exitCode = v
		case string:
			code, err := strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("exit code must be an integer, got %v", v)
			}
			exitCode = code
		default:
			return nil, fmt.Errorf("exit code must be an integer, got %T", v)
		}
		return map[string]interface{}{
			"exit": true,
			"code": exitCode,
		}, nil
	}

	// If more than one argument, first argument must be a destination
	// and second argument must be an integer
	if len(args) >= 2 {
		// Try to convert the second argument to an integer
		switch v := args[1].(type) {
		case int:
			exitCode = v
		case string:
			code, err := strconv.Atoi(v)
			if err != nil {
				return nil, fmt.Errorf("exit code must be an integer, got %v", v)
			}
			exitCode = code
		default:
			return nil, fmt.Errorf("exit code must be an integer, got %T", v)
		}
		return map[string]interface{}{
			"exit": true,
			"code": exitCode,
		}, nil
	}

	return nil, nil
}
