package parsers

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/loggers"
)

// MapContextProperty maps a token to its value in the context, if it exists
func MapContextProperty(source string, slots map[string]interface{}, arg interface{}) (interface{}, error) {

	// Handle nested arrays
	if arr, ok := arg.([]interface{}); ok {
		resolvedArr := make([]interface{}, 0, len(arr))
		for _, elem := range arr {

			// Handle property references
			if ref, ok1 := GetPropertyRef(elem); ok1 {
				if val, ok2 := slots[ref.Name]; ok2 {
					loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Mapped as PropertyRef: %v", source, val)

					if ref.Spread {

						// If the property is a slice, spread it into the array
						if slice, ok3 := val.([]interface{}); ok3 {
							resolvedArr = append(resolvedArr, slice...)
							continue
						}

						// If the property is not a slice, just append it
					}

					resolvedArr = append(resolvedArr, val)
					continue
				}
				return nil, fmt.Errorf("[%s]: undefined property: %s", source, ref.Name)
			}

			val, err := MapContextProperty(source, slots, elem)
			if err != nil {
				return nil, err
			}
			resolvedArr = append(resolvedArr, val)
		}
		loggers.Printf(loggers.Debug, "[%s]: MapContextProperty: Mapped as array: %v", source, resolvedArr)
		return resolvedArr, nil
	}

	// Handle property references
	if ref, ok := GetPropertyRef(arg); ok {
		if val, ok2 := slots[ref.Name]; ok2 {
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
	resolvedArgs := make([]interface{}, 0, len(args))
	for _, arg := range args {

		// Handle property references
		if ref, ok1 := GetPropertyRef(arg); ok1 {
			if val, ok2 := slots[ref.Name]; ok2 {
				loggers.Printf(loggers.Debug, "[%s]: MapContextProperties: Mapped as PropertyRef: %v", source, val)

				if ref.Spread {

					// If the property is a slice, spread it into the array
					if slice, ok3 := val.([]interface{}); ok3 {
						resolvedArgs = append(resolvedArgs, slice...)
						continue
					}

					// If the property is not a slice, just append it
				}

				resolvedArgs = append(resolvedArgs, val)
				continue
			}
			return nil, fmt.Errorf("[%s]: undefined property: %s", source, ref.Name)
		}

		val, err := MapContextProperty(source, slots, arg)
		if err != nil {
			return nil, fmt.Errorf("[%s]: MapContextProperties: argument: %v", source, err)
		}
		resolvedArgs = append(resolvedArgs, val)
	}
	loggers.Printf(loggers.Debug, "[%s]: MapContextProperties: Mapped args: %v to %v", source, args, resolvedArgs)
	return resolvedArgs, nil
}
