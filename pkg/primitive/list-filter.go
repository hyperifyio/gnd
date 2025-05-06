package primitive

import (
	"encoding/json"
	"fmt"
)

type listFilterPrimitive struct{}

func (p *listFilterPrimitive) Name() string {
	return "/gnd/list-filter"
}

func (p *listFilterPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("list-filter expects 2 arguments")
	}
	return ListFilter(args[0], args[1])
}

// ListFilter filters a list based on a predicate function
func ListFilter(list, fn string) ([]interface{}, error) {
	// Parse the list
	var items []interface{}
	if err := json.Unmarshal([]byte(list), &items); err != nil {
		return nil, fmt.Errorf("invalid list: %v", err)
	}

	// Apply the predicate to each element
	var result []interface{}
	for _, item := range items {
		// Convert item to string
		itemStr, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("failed to convert item to string: %v", err)
		}

		// Call the predicate
		prim, ok := Get(fn)
		if !ok {
			return nil, fmt.Errorf("unknown function: %s", fn)
		}

		// Execute the predicate
		val, err := prim.Execute([]string{string(itemStr)})
		if err != nil {
			return nil, fmt.Errorf("predicate execution failed: %v", err)
		}

		// Check if the predicate returned true
		if b, ok := val.(bool); ok && b {
			result = append(result, item)
		}
	}

	return result, nil
}

func init() {
	RegisterPrimitive(&listFilterPrimitive{})
}
