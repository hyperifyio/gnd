package primitive

import (
	"encoding/json"
	"fmt"
	"reflect"
)

type dictPrimitive struct{}

func (p *dictPrimitive) Name() string {
	return "/gnd/dict"
}

func (p *dictPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("dict expects 1 argument")
	}
	return Dict(args[0])
}

// Dict creates a dictionary from a JSON string
func Dict(jsonStr string) (map[string]interface{}, error) {
	// Parse the JSON string
	var dict map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &dict); err != nil {
		return nil, fmt.Errorf("invalid dictionary: %v", err)
	}
	return dict, nil
}

func init() {
	RegisterPrimitive(&dictPrimitive{})
}

// DictGet gets a value from a dictionary by key
func DictGet(dict interface{}, key interface{}) (interface{}, error) {
	v := reflect.ValueOf(dict)
	if v.Kind() != reflect.Map {
		return nil, fmt.Errorf("input is not a dictionary: %v", v.Kind())
	}

	k := reflect.ValueOf(key)
	if !k.Type().AssignableTo(v.Type().Key()) {
		return nil, fmt.Errorf("key type mismatch: expected %v, got %v", v.Type().Key(), k.Type())
	}

	val := v.MapIndex(k)
	if !val.IsValid() {
		return nil, fmt.Errorf("key not found: %v", key)
	}

	return val.Interface(), nil
}

// DictSet sets a value in a dictionary by key
func DictSet(dict interface{}, key interface{}, value interface{}) (interface{}, error) {
	v := reflect.ValueOf(dict)
	if v.Kind() != reflect.Map {
		return nil, fmt.Errorf("input is not a dictionary: %v", v.Kind())
	}

	k := reflect.ValueOf(key)
	if !k.Type().AssignableTo(v.Type().Key()) {
		return nil, fmt.Errorf("key type mismatch: expected %v, got %v", v.Type().Key(), k.Type())
	}

	val := reflect.ValueOf(value)
	if !val.Type().AssignableTo(v.Type().Elem()) {
		return nil, fmt.Errorf("value type mismatch: expected %v, got %v", v.Type().Elem(), val.Type())
	}

	// Create a new map to preserve immutability
	newMap := reflect.MakeMap(v.Type())
	iter := v.MapRange()
	for iter.Next() {
		newMap.SetMapIndex(iter.Key(), iter.Value())
	}
	newMap.SetMapIndex(k, val)

	return newMap.Interface(), nil
}
