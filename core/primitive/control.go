package primitive

import (
	"fmt"
	"reflect"
)

// Iterate walks a list calling a function token
func Iterate(fnToken string, acc interface{}, list interface{}, limit int) (interface{}, error) {
	v := reflect.ValueOf(list)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		return nil, fmt.Errorf("input is not a list: %v", v.Kind())
	}

	fn, err := getFunction(fnToken)
	if err != nil {
		return nil, fmt.Errorf("invalid function token %q: %w", fnToken, err)
	}

	result := acc
	for i := 0; i < v.Len() && i < limit; i++ {
		args := []reflect.Value{reflect.ValueOf(result), v.Index(i)}
		results := fn.Call(args)

		if len(results) > 1 && !results[1].IsNil() {
			return nil, results[1].Interface().(error)
		}

		result = results[0].Interface()
	}

	return result, nil
}

// Select chooses between two values based on a condition
func Select(cond bool, thenVal, elseVal interface{}) interface{} {
	if cond {
		return thenVal
	}
	return elseVal
}

// Identity returns its input unchanged
func Identity(x interface{}) interface{} {
	return x
}

// MakeError creates an error with the given message
func MakeError(msg string) error {
	return fmt.Errorf(msg)
} 