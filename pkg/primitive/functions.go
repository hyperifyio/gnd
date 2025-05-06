package primitive

import (
	"fmt"
	"reflect"
)

// getFunction returns a function from the registry
func getFunction(token string) (reflect.Value, error) {
	fn, exists := Get(token)
	if !exists {
		return reflect.Value{}, fmt.Errorf("unknown function token: %s", token)
	}
	return reflect.ValueOf(fn.Execute), nil
}

func init() {
	// Only register identity and llm primitives
	RegisterPrimitive(&Identity{})
	RegisterPrimitive(&LLM{})
}

// IsEven checks if a number is even
type IsEven struct{}

func (i *IsEven) Name() string {
	return "/gnd/is_even"
}

func (i *IsEven) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("is_even expects 1 argument, got %d", len(args))
	}
	num, err := ParseNumber(args[0])
	if err != nil {
		return false, fmt.Errorf("expected number, got %v", err)
	}
	switch v := num.(type) {
	case int64:
		return v%2 == 0, nil
	case float64:
		return int64(v)%2 == 0, nil
	default:
		return false, fmt.Errorf("expected number, got %T", v)
	}
}

// IsLong checks if a string is longer than 2 characters
type IsLong struct{}

func (i *IsLong) Name() string {
	return "/gnd/is_long"
}

func (i *IsLong) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("is_long expects 1 argument, got %d", len(args))
	}
	return len(args[0]) > 2, nil
}

// Inc increments a number by 1
type Inc struct{}

func (i *Inc) Name() string {
	return "/gnd/inc"
}

func (i *Inc) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("inc expects 1 argument, got %d", len(args))
	}
	num, err := ParseNumber(args[0])
	if err != nil {
		return nil, fmt.Errorf("expected number, got %v", err)
	}
	switch v := num.(type) {
	case int64:
		return v + 1, nil
	case float64:
		return v + 1, nil
	default:
		return nil, fmt.Errorf("expected number, got %T", v)
	}
}

// Double multiplies a number by 2
type Double struct{}

func (d *Double) Name() string {
	return "/gnd/double"
}

func (d *Double) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("double expects 1 argument, got %d", len(args))
	}
	num, err := ParseNumber(args[0])
	if err != nil {
		return nil, fmt.Errorf("expected number, got %v", err)
	}
	switch v := num.(type) {
	case int64:
		return v * 2, nil
	case float64:
		return v * 2, nil
	default:
		return nil, fmt.Errorf("expected number, got %T", v)
	}
}
