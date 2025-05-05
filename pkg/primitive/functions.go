package primitive

import (
	"fmt"
	"reflect"
)

// getFunction returns a function from a token
func getFunction(token string) (reflect.Value, error) {
	switch token {
	case "identity":
		return reflect.ValueOf(func(x interface{}) (interface{}, error) {
			return x, nil
		}), nil
	case "is_even":
		return reflect.ValueOf(func(x interface{}) (bool, error) {
			v := reflect.ValueOf(x)
			if v.Kind() != reflect.Int && v.Kind() != reflect.Int64 {
				return false, fmt.Errorf("expected int, got %v", v.Kind())
			}
			return v.Int()%2 == 0, nil
		}), nil
	case "is_long":
		return reflect.ValueOf(func(x interface{}) (bool, error) {
			v := reflect.ValueOf(x)
			if v.Kind() != reflect.String {
				return false, fmt.Errorf("expected string, got %v", v.Kind())
			}
			return len(v.String()) > 2, nil
		}), nil
	case "add":
		return reflect.ValueOf(func(x, y interface{}) (interface{}, error) {
			vx := reflect.ValueOf(x)
			vy := reflect.ValueOf(y)
			if (vx.Kind() != reflect.Int && vx.Kind() != reflect.Int64) || (vy.Kind() != reflect.Int && vy.Kind() != reflect.Int64) {
				return nil, fmt.Errorf("expected ints, got %v and %v", vx.Kind(), vy.Kind())
			}
			return int(vx.Int() + vy.Int()), nil
		}), nil
	case "concat":
		return reflect.ValueOf(func(x, y interface{}) (interface{}, error) {
			vx := reflect.ValueOf(x)
			vy := reflect.ValueOf(y)
			if vx.Kind() != reflect.String || vy.Kind() != reflect.String {
				return nil, fmt.Errorf("expected strings, got %v and %v", vx.Kind(), vy.Kind())
			}
			return vx.String() + vy.String(), nil
		}), nil
	case "inc":
		return reflect.ValueOf(func(acc, x interface{}) (interface{}, error) {
			vacc := reflect.ValueOf(acc)
			if vacc.Kind() != reflect.Int && vacc.Kind() != reflect.Int64 {
				return nil, fmt.Errorf("expected int accumulator, got %v", vacc.Kind())
			}
			return int(vacc.Int() + 1), nil
		}), nil
	case "double":
		return reflect.ValueOf(func(acc, x interface{}) (interface{}, error) {
			vacc := reflect.ValueOf(acc)
			if vacc.Kind() != reflect.Int && vacc.Kind() != reflect.Int64 {
				return nil, fmt.Errorf("expected int accumulator, got %v", vacc.Kind())
			}
			return int(vacc.Int() * 2), nil
		}), nil
	default:
		return reflect.Value{}, fmt.Errorf("unknown function token: %s", token)
	}
} 