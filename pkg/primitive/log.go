package primitive

import (
	"fmt"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
)

// Log levels
const (
	LogError = iota
	LogWarn
	LogInfo
	LogDebug
)

// Log represents the log primitive
type Log struct{}

// Name returns the name of the primitive
func (l *Log) Name() string {
	return "/gnd/log"
}

// Execute runs the log primitive
func (l *Log) Execute(args []interface{}) (interface{}, error) {
	// Case 1: No arguments - use _ as an array
	if len(args) == 0 {
		log.Printf(log.Info, "_")
		return nil, nil
	}

	// Case 2: Single argument - could be a string or an array from _
	if len(args) == 1 {
		// Check if it's an array
		if arr, ok := args[0].([]interface{}); ok {
			// First element should be the log level
			levelStr, ok := arr[0].(string)
			if !ok {
				return nil, fmt.Errorf("log level must be a string, got %T", arr[0])
			}

			// Convert level string to int
			var level int
			normalizedLevel := strings.ToLower(levelStr)
			switch normalizedLevel {
			case "error":
				level = log.Error
			case "warn":
				level = log.Warn
			case "info":
				level = log.Info
			case "debug":
				level = log.Debug
			default:
				return nil, fmt.Errorf("invalid log level: %s (must be one of: error, warn, info, debug)", levelStr)
			}

			// Convert remaining elements to strings
			var strArgs []string
			for _, item := range arr[1:] {
				str, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("log requires string arguments, got %T", item)
				}
				strArgs = append(strArgs, str)
			}

			// Join all arguments with spaces and log
			output := strings.Join(strArgs, " ")
			log.Printf(level, output)

			// Return the last argument
			if len(arr) > 1 {
				return arr[len(arr)-1], nil
			}
			return arr[0], nil
		}

		// If it's a string, log at info level
		val, ok := args[0].(string)
		if !ok {
			return nil, fmt.Errorf("log requires string arguments, got %T", args[0])
		}
		log.Printf(log.Info, val)
		return val, nil
	}

	// Case 3: Level and value(s) provided
	levelStr, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("log level must be a string, got %T", args[0])
	}

	// Convert level string to int
	var level int
	normalizedLevel := strings.ToLower(levelStr)
	switch normalizedLevel {
	case "error":
		level = log.Error
	case "warn":
		level = log.Warn
	case "info":
		level = log.Info
	case "debug":
		level = log.Debug
	default:
		return nil, fmt.Errorf("invalid log level: %s (must be one of: error, warn, info, debug)", levelStr)
	}

	// Convert remaining arguments to strings, handling arrays
	var strArgs []string
	for _, arg := range args[1:] {
		switch v := arg.(type) {
		case string:
			strArgs = append(strArgs, v)
		case []interface{}:
			// For arrays, convert each element to string
			for _, item := range v {
				str, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("log requires string arguments, got %T", item)
				}
				strArgs = append(strArgs, str)
			}
		default:
			// Try to convert the argument to a string
			str, ok := arg.(string)
			if !ok {
				return nil, fmt.Errorf("log requires string arguments, got %T", arg)
			}
			strArgs = append(strArgs, str)
		}
	}

	// Join all arguments with spaces and log
	output := strings.Join(strArgs, " ")
	log.Printf(level, output)

	// Return the last argument
	if len(args) > 1 {
		if arr, ok := args[len(args)-1].([]interface{}); ok && len(arr) > 0 {
			return arr[len(arr)-1], nil
		}
		return args[len(args)-1], nil
	}
	return args[0], nil
}

func init() {
	RegisterPrimitive(&Log{})
}
