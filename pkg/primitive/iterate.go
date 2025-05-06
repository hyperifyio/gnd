package primitive

import (
	"encoding/json"
	"fmt"
)

type iteratePrimitive struct{}

func (p *iteratePrimitive) Name() string {
	return "/gnd/iterate"
}

func (p *iteratePrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 4 {
		return nil, fmt.Errorf("iterate expects 4 arguments")
	}
	return IterateList(args[0], args[1], args[2], args[3])
}

// IterateList applies a function to each element in a list
func IterateList(fn, acc, list, limit string) (interface{}, error) {
	// Parse the list
	var items []interface{}
	if err := json.Unmarshal([]byte(list), &items); err != nil {
		return nil, fmt.Errorf("invalid list: %v", err)
	}

	// Parse the initial value
	var result interface{}
	if err := json.Unmarshal([]byte(acc), &result); err != nil {
		return nil, fmt.Errorf("invalid initial value: %v", err)
	}

	// Parse the limit
	var limitNum int
	if err := json.Unmarshal([]byte(limit), &limitNum); err != nil {
		return nil, fmt.Errorf("invalid limit: %v", err)
	}

	// Apply the function to each element
	for i := 0; i < len(items) && i < limitNum; i++ {
		// Convert accumulator to string
		accStr, err := json.Marshal(result)
		if err != nil {
			return nil, fmt.Errorf("failed to convert accumulator to string: %v", err)
		}

		// Convert item to string
		itemStr, err := json.Marshal(items[i])
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

		result = val
	}

	return result, nil
}

func init() {
	RegisterPrimitive(&iteratePrimitive{})
}
