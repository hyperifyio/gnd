package parsers

import (
	"encoding/json"
	"fmt"
	"strings"
)

func ParseString(arg interface{}) (string, error) {
	switch v := arg.(type) {

	case nil:
		return "nil", nil

	case string:
		return v, nil

	case int:
		return fmt.Sprintf("%d", v), nil

	case bool:
		return fmt.Sprintf("%t", v), nil

	case float64:
		return fmt.Sprintf("%v", v), nil

	case float32:
		return fmt.Sprintf("%v", v), nil

	case map[string]interface{}:
		b, err := json.Marshal(v)
		if err != nil {
			return "", err
		}
		return string(b), nil

	case []interface{}:

		// For arrays, convert each element to string
		var strArgs []string
		for _, item := range v {
			str, err := ParseString(item)
			if err != nil {
				return "", fmt.Errorf("invalid: %v: %v", item, err)
			}
			strArgs = append(strArgs, str)
		}
		return strings.Join(strArgs, " "), nil

	default:
		return "", fmt.Errorf("invalid type: %T", arg)
	}
}
