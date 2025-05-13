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

// ConvertLogLevel converts a string log level to its corresponding integer value
func ConvertLogLevel(levelStr string) (int, error) {
	normalizedLevel := strings.ToLower(levelStr)
	switch normalizedLevel {
	case "error":
		return log.Error, nil
	case "warn":
		return log.Warn, nil
	case "info":
		return log.Info, nil
	case "debug":
		return log.Debug, nil
	default:
		return 0, fmt.Errorf("invalid log level: %s (must be one of: error, warn, info, debug)", levelStr)
	}
}

// ConvertToStrings converts log arguments to a slice of strings
func ConvertToStrings(args []interface{}) ([]string, error) {
	var strArgs []string
	for _, arg := range args {
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
	return strArgs, nil
}

// Log represents the log primitive
type Log struct{}

// Name returns the name of the primitive
func (l *Log) Name() string {
	return "/gnd/log"
}

// Execute runs the log primitive
func (l *Log) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 || len(args) == 1 {
		return nil, fmt.Errorf("log primitive requires at least two arguments")
	}

	// Case 3: Level and value(s) provided
	levelStr, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("log level must be a string, got %T", args[0])
	}

	// Convert level string to int
	level, err := ConvertLogLevel(levelStr)
	if err != nil {
		return nil, err
	}

	// Convert remaining arguments to strings
	strArgs, err := ConvertToStrings(args[1:])
	if err != nil {
		return nil, err
	}

	// Join all arguments with spaces and log
	output := strings.Join(strArgs, " ")
	log.Printf(level, output)
	// Return the last string argument, as expected by the tests
	if len(strArgs) > 0 {
		return strArgs[len(strArgs)-1], nil
	}
	return output, nil
}

func init() {
	RegisterPrimitive(&Log{})
}
