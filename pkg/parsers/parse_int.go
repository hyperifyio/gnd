package parsers

import (
	"fmt"
	"strconv"
)

// ParseInt tries to convert the argument to an integer
func ParseInt(arg interface{}) (int, error) {
	switch v := arg.(type) {
	case int:
		return v, nil
	case string:
		code, err := strconv.Atoi(v)
		if err != nil {
			return 0, fmt.Errorf("ParseInt: value invalid: %v", v)
		}
		return code, nil
	default:
		return 0, fmt.Errorf("ParseInt: value must be an integer: %T", v)
	}
}
