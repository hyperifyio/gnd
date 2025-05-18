package parsers

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

var (
	ErrInvalidArrayElement = errors.New("invalid array element")
	ErrFailedToMarshalMap  = errors.New("failed to marshal map")
	ErrUnsupportedType     = errors.New("unsupported type")
)

// needsEscaping returns true if the string contains characters that need escaping
func needsEscaping(s string) bool {
	return strings.ContainsAny(s, " \t\n\r\"'\\")
}

// escapeString properly escapes a string and wraps it in quotes if needed
func escapeString(s string) string {
	if !needsEscaping(s) {
		return s
	}
	b, _ := json.Marshal(s)
	return string(b)
}

func ParseString(arg interface{}) (string, error) {
	switch v := arg.(type) {
	case nil:
		return "nil", nil

	case string:
		return escapeString(v), nil

	case int, int8, int16, int32, int64:
		return fmt.Sprintf("%d", v), nil

	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", v), nil

	case float32, float64:
		return fmt.Sprintf("%v", v), nil

	case bool:
		return fmt.Sprintf("%t", v), nil

	case map[string]interface{}:
		if len(v) == 0 {
			return "{}", nil
		}
		// Get sorted keys
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		var pairs []string
		for _, key := range keys {
			valStr, err := ParseString(v[key])
			if err != nil {
				return "", ErrFailedToMarshalMap
			}
			pairs = append(pairs, key, valStr)
		}
		return "{ " + strings.Join(pairs, " ") + " }", nil

	case []interface{}:
		if len(v) == 0 {
			return "[]", nil
		}
		var strArgs []string
		for _, item := range v {
			str, err := ParseString(item)
			if err != nil {
				return "", ErrInvalidArrayElement
			}
			strArgs = append(strArgs, str)
		}
		return "[ " + strings.Join(strArgs, " ") + " ]", nil

	case []string:
		if len(v) == 0 {
			return "[]", nil
		}
		var strArgs []string
		for _, item := range v {
			str, err := ParseString(item)
			if err != nil {
				return "", ErrInvalidArrayElement
			}
			strArgs = append(strArgs, str)
		}
		return "[ " + strings.Join(strArgs, " ") + " ]", nil

	default:
		// Try to handle other numeric types through reflection
		rv := reflect.ValueOf(arg)
		switch rv.Kind() {

		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return fmt.Sprintf("%d", rv.Int()), nil

		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			return fmt.Sprintf("%d", rv.Uint()), nil

		case reflect.Float32, reflect.Float64:
			return fmt.Sprintf("%v", rv.Float()), nil

		case reflect.Slice:
			if rv.Len() == 0 {
				return "[]", nil
			}
			var strArgs []string
			for i := 0; i < rv.Len(); i++ {
				str, err := ParseString(rv.Index(i).Interface())
				if err != nil {
					return "", ErrInvalidArrayElement
				}
				strArgs = append(strArgs, str)
			}
			return "[ " + strings.Join(strArgs, " ") + " ]", nil

		case reflect.Map:
			if rv.Len() == 0 {
				return "{}", nil
			}
			// Get sorted keys
			keys := make([]string, 0, rv.Len())
			iter := rv.MapRange()
			for iter.Next() {
				key := iter.Key()
				keyStr, err := ParseString(key.Interface())
				if err != nil {
					return "", ErrFailedToMarshalMap
				}
				keys = append(keys, keyStr)
			}
			sort.Strings(keys)

			var pairs []string
			for _, keyStr := range keys {
				// Find the value for this key
				iter := rv.MapRange()
				for iter.Next() {
					key := iter.Key()
					keyStr2, _ := ParseString(key.Interface())
					if keyStr2 == keyStr {
						valStr, err := ParseString(iter.Value().Interface())
						if err != nil {
							return "", ErrFailedToMarshalMap
						}
						pairs = append(pairs, keyStr, valStr)
						break
					}
				}
			}
			return "{ " + strings.Join(pairs, " ") + " }", nil
		}
		return "", ErrUnsupportedType
	}
}
