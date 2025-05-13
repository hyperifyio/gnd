package primitive

import "fmt"

// ReturnValue represents a return value from `return` operation
type ReturnValue struct {
	Value interface{}
}

func (e *ReturnValue) Error() string {
	return fmt.Sprintf("return with %v", e.Value)
}

// NewReturnValue creates a new ReturnValue with the given code and optional value
func NewReturnValue(value interface{}) *ReturnValue {
	return &ReturnValue{
		Value: value,
	}
}

// GetReturnValue extracts the ReturnValue from a value if it is one
func GetReturnValue(v interface{}) (*ReturnValue, bool) {
	result, ok := v.(*ReturnValue)
	return result, ok
}
