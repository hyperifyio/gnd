package primitive

import (
	"fmt"
	"os"
	"strings"
)

// Print represents the print primitive
type Print struct{}

// Name returns the name of the primitive
func (p *Print) Name() string {
	return "/gnd/print"
}

// Execute runs the print primitive
func (p *Print) Execute(args []interface{}) (interface{}, error) {
	// Convert all arguments to strings, handling arrays
	var strArgs []string
	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			strArgs = append(strArgs, v)
		case []interface{}:
			// For arrays, convert each element to string
			for _, item := range v {
				str, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("print requires string arguments, got %T", item)
				}
				strArgs = append(strArgs, str)
			}
		default:
			// Try to convert the argument to a string
			str, ok := arg.(string)
			if !ok {
				return nil, fmt.Errorf("print requires string arguments, got %T", arg)
			}
			strArgs = append(strArgs, str)
		}
	}

	// Join all arguments with spaces and print
	output := strings.Join(strArgs, " ")
	fmt.Fprintln(os.Stdout, output)

	// Return the last argument
	if len(args) > 0 {
		if arr, ok := args[len(args)-1].([]interface{}); ok && len(arr) > 0 {
			return arr[len(arr)-1], nil
		}
		return args[len(args)-1], nil
	}
	return nil, nil
}
