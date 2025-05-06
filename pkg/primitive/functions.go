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
