package primitive

import (
	"encoding/json"
	"fmt"
)

type listMapPrimitive struct{}

func (p *listMapPrimitive) Name() string {
	return "/gnd/list-map"
}

func (p *listMapPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("list-map expects 2 arguments")
	}
	return ListMap(args[0], args[1])
}

// ListMap applies a function to each element in a list
func ListMap(list, fn string) ([]interface{}, error) {
	// Parse the list
	var items []interface{}
	if err := json.Unmarshal([]byte(list), &items); err != nil {
		return nil, fmt.Errorf("invalid list: %v", err)
	}

	// Apply the function to each element
	result := make([]interface{}, len(items))
	for i, item := range items {
		// Convert item to string
		itemStr, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("failed to convert item to string: %v", err)
		}

		// Call the function
		prim, ok := Get(fn)
		if !ok {
			return nil, fmt.Errorf("unknown function: %s", fn)
		}

		// Execute the function
		val, err := prim.Execute([]string{string(itemStr)})
		if err != nil {
			return nil, fmt.Errorf("function execution failed: %v", err)
		}

		result[i] = val
	}

	return result, nil
}

func init() {
	RegisterPrimitive(&listMapPrimitive{})
}
