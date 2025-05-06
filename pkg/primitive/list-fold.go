package primitive

import (
	"encoding/json"
	"fmt"
)

type listFoldPrimitive struct{}

func (p *listFoldPrimitive) Name() string {
	return "/gnd/list-fold"
}

func (p *listFoldPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("list-fold expects 3 arguments")
	}
	return ListFold(args[0], args[1], args[2])
}

// ListFold reduces a list to a single value using a function
func ListFold(list, init, fn string) (interface{}, error) {
	// Parse the list
	var items []interface{}
	if err := json.Unmarshal([]byte(list), &items); err != nil {
		return nil, fmt.Errorf("invalid list: %v", err)
	}

	// Parse the initial value
	var acc interface{}
	if err := json.Unmarshal([]byte(init), &acc); err != nil {
		return nil, fmt.Errorf("invalid initial value: %v", err)
	}

	// Apply the function to each element
	for _, item := range items {
		// Convert accumulator to string
		accStr, err := json.Marshal(acc)
		if err != nil {
			return nil, fmt.Errorf("failed to convert accumulator to string: %v", err)
		}

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
		val, err := prim.Execute([]string{string(accStr), string(itemStr)})
		if err != nil {
			return nil, fmt.Errorf("function execution failed: %v", err)
		}

		acc = val
	}

	return acc, nil
}

func init() {
	RegisterPrimitive(&listFoldPrimitive{})
}
