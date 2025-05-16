package primitive

import (
	"errors"
	"fmt"
	"strings"
)

var ConcatExpectsAtLeastOneArgument = errors.New("concat expects at least 1 argument, got 0")

// Concat represents the concat primitive
type Concat struct{}

func (c *Concat) Name() string {
	return "/gnd/concat"
}

func (c *Concat) Execute(args []interface{}) (interface{}, error) {

	if len(args) == 0 {
		return nil, ConcatExpectsAtLeastOneArgument
	}

	// If only one argument, return it directly
	if len(args) == 1 {
		return args[0], nil
	}

	// Check if first argument is an array
	if _, ok := args[0].([]interface{}); ok {
		return c.concatArrays(args)
	}

	// Otherwise treat as string concatenation
	return c.concatStrings(args)
}

// concatArrays concatenates arrays and non-array values
func (c *Concat) concatArrays(args []interface{}) (interface{}, error) {
	var result []interface{}

	// Process each argument
	for _, arg := range args {
		if arr, ok := arg.([]interface{}); ok {
			result = append(result, arr...)
		} else if str, ok := arg.(string); ok {
			result = append(result, str)
		} else {
			result = append(result, arg)
		}
	}

	return result, nil
}

// concatStrings concatenates all arguments as strings
func (c *Concat) concatStrings(args []interface{}) (interface{}, error) {
	var result strings.Builder
	for _, arg := range args {
		result.WriteString(fmt.Sprintf("%v", arg))
	}
	return result.String(), nil
}

func init() {
	RegisterPrimitive(&Concat{})
}
