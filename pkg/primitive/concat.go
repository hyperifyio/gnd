package primitive

import (
	"fmt"
	"reflect"
)

type concatPrimitive struct{}

func (p *concatPrimitive) Name() string {
	return "/gnd/concat"
}

func (p *concatPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("concat expects 2 arguments: two strings to concatenate")
	}
	return Concat(args[0], args[1])
}

// Concat concatenates two strings or two arrays
func Concat(a, b interface{}) (interface{}, error) {
	va := reflect.ValueOf(a)
	vb := reflect.ValueOf(b)

	if va.Kind() != vb.Kind() {
		return nil, fmt.Errorf("cannot concat different types: %v and %v", va.Kind(), vb.Kind())
	}

	switch va.Kind() {
	case reflect.String:
		return va.String() + vb.String(), nil

	case reflect.Slice, reflect.Array:
		if va.Type().Elem() != vb.Type().Elem() {
			return nil, fmt.Errorf("cannot concat arrays of different element types: %v and %v", va.Type().Elem(), vb.Type().Elem())
		}

		result := reflect.MakeSlice(va.Type(), va.Len()+vb.Len(), va.Len()+vb.Len())
		reflect.Copy(result, va)
		reflect.Copy(result.Slice(va.Len(), result.Len()), vb)
		return result.Interface(), nil

	default:
		return nil, fmt.Errorf("cannot concat type: %v", va.Kind())
	}
}

func init() {
	RegisterPrimitive(&concatPrimitive{})
}
