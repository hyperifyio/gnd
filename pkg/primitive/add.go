package primitive

import (
	"fmt"
)

type addPrimitive struct{}

func (p *addPrimitive) Name() string {
	return "/gnd/add"
}

func (p *addPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("add expects 2 arguments")
	}
	return AddNumbers(args[0], args[1])
}

// AddNumbers adds two numbers
func AddNumbers(a, b string) (float64, error) {
	// Parse first number
	num1, err := ParseNumber(a)
	if err != nil {
		return 0, fmt.Errorf("first argument is not a number: %v", err)
	}

	// Parse second number
	num2, err := ParseNumber(b)
	if err != nil {
		return 0, fmt.Errorf("second argument is not a number: %v", err)
	}

	// Convert to float64
	var f1, f2 float64
	switch v := num1.(type) {
	case int64:
		f1 = float64(v)
	case float64:
		f1 = v
	default:
		return 0, fmt.Errorf("first argument is not a number")
	}

	switch v := num2.(type) {
	case int64:
		f2 = float64(v)
	case float64:
		f2 = v
	default:
		return 0, fmt.Errorf("second argument is not a number")
	}

	// Add the numbers
	return f1 + f2, nil
}

func init() {
	RegisterPrimitive(&addPrimitive{})
}
