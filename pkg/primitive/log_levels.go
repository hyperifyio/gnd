package primitive

import (
	"fmt"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
)

// LogLevel represents a log level primitive
type LogLevel struct {
	level int
}

// Name returns the name of the primitive
func (l *LogLevel) Name() string {
	switch l.level {
	case log.Error:
		return "/gnd/error"
	case log.Warn:
		return "/gnd/warn"
	case log.Info:
		return "/gnd/info"
	case log.Debug:
		return "/gnd/debug"
	default:
		return "/gnd/log"
	}
}

// Execute runs the log level primitive
func (l *LogLevel) Execute(args []interface{}) (interface{}, error) {
	// Convert arguments to strings, handling arrays
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

	// Join all arguments with spaces and log
	output := strings.Join(strArgs, " ")
	log.Printf(l.level, output)

	// Return the last argument
	if len(args) > 0 {
		if arr, ok := args[len(args)-1].([]interface{}); ok && len(arr) > 0 {
			return arr[len(arr)-1], nil
		}
		return args[len(args)-1], nil
	}
	return nil, nil
}

func init() {
	// Register log level primitives
	Register(&LogLevel{level: log.Error})
	Register(&LogLevel{level: log.Warn})
	Register(&LogLevel{level: log.Info})
	Register(&LogLevel{level: log.Debug})
}
