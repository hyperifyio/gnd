package primitives

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

// String returns the string representation of the ReturnValue
func (r *ReturnValue) String() string {
	return fmt.Sprintf("ReturnValue{%v}", r.Value)
}

// Format formats the ReturnValue for printing
func (c *ReturnValue) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v':
		if f.Flag('+') {
			fmt.Fprintf(f, "ReturnValue{Value: %+v}", c.Value)
			return
		}
		fallthrough
	default:
		fmt.Fprint(f, c.String())
	}
}
