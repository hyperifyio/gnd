package parsers

import "fmt"

// PropertyRef is a wrapper for property reference tokens
type PropertyRef struct {
	Name string
}

// MapContextProperty maps a token to its value in the context, if it exists
func MapContextProperty(slots map[string]interface{}, arg interface{}) (interface{}, error) {
	if ref, ok := arg.(PropertyRef); ok {
		if val, ok := slots[ref.Name]; ok {
			return val, nil
		}
		return nil, fmt.Errorf("undefined property: %s", ref.Name)
	}
	// Handle nested arrays
	if arr, ok := arg.([]interface{}); ok {
		resolvedArr := make([]interface{}, len(arr))
		for i, elem := range arr {
			val, err := MapContextProperty(slots, elem)
			if err != nil {
				return nil, err
			}
			resolvedArr[i] = val
		}
		return resolvedArr, nil
	}
	// Handle nested maps
	if m, ok := arg.(map[string]interface{}); ok {
		resolvedMap := make(map[string]interface{})
		for k, v := range m {
			val, err := MapContextProperty(slots, v)
			if err != nil {
				return nil, err
			}
			resolvedMap[k] = val
		}
		return resolvedMap, nil
	}
	// Not a property reference, just return as-is
	return arg, nil
}

// MapContextProperties maps an array of tokens to their values in the context
func MapContextProperties(slots map[string]interface{}, args []interface{}) ([]interface{}, error) {
	resolvedArgs := make([]interface{}, len(args))
	for j, arg := range args {
		val, err := MapContextProperty(slots, arg)
		if err != nil {
			return nil, fmt.Errorf("argument %d: %v", j, err)
		}
		resolvedArgs[j] = val
	}
	return resolvedArgs, nil
}
