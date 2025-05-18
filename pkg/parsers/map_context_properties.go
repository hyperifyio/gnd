package parsers

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// MapContextProperty maps a token to its value in the context, if it exists
func MapContextProperty(source string, slots map[string]interface{}, arg interface{}) (interface{}, error) {

	// Handle nested arrays
	if arr, ok := arg.([]interface{}); ok {
		resolvedArr := make([]interface{}, len(arr))
		for i, elem := range arr {
			val, err := MapContextProperty(source, slots, elem)
			if err != nil {
				return nil, err
			}
			resolvedArr[i] = val
		}
		loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Mapped as array: %v", source, resolvedArr)
		return resolvedArr, nil
	}

	// Handle property references
	if ref, ok := GetPropertyRef(arg); ok {
		if val, ok := slots[ref.Name]; ok {
			loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Mapped as PropertyRef: %v", source, val)
			return val, nil
		}
		return nil, fmt.Errorf("[%s]: undefined property: %s", source, ref.Name)
	}

	// Handle nested maps
	if m, ok := arg.(map[string]interface{}); ok {
		resolvedMap := make(map[string]interface{})
		for k, v := range m {
			val, err := MapContextProperty(source, slots, v)
			if err != nil {
				return nil, err
			}
			resolvedMap[k] = val
		}
		loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Mapped as map: %v", source, resolvedMap)
		return resolvedMap, nil
	}

	// Not a property reference, just return as-is
	loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Not a property reference, mapped as: %s", source, arg)
	return arg, nil
}

// MapContextProperties maps an array of tokens to their values in the context
func MapContextProperties(source string, slots map[string]interface{}, args []interface{}) ([]interface{}, error) {
	resolvedArgs := make([]interface{}, len(args))
	for j, arg := range args {
		val, err := MapContextProperty(source, slots, arg)
		if err != nil {
			return nil, fmt.Errorf("[%s]: MapContextProperties: argument %d: %v", source, j, err)
		}
		resolvedArgs[j] = val
	}
	loggers.Printf(loggers.Debug, "[%s]: MapContextProperties: Mapped args: %v to %v", source, args, resolvedArgs)
	return resolvedArgs, nil
}
