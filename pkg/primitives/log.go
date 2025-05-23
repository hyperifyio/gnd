package primitives

import (
	"errors"
	"fmt"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"strings"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

var LogPrimitiveRequiresAtLeastTwoArguments = errors.New("log: primitive requires at least two arguments")

// ConvertLogLevel converts a string log level to its corresponding integer value
func ConvertLogLevel(levelStr string) (int, error) {
	normalizedLevel := strings.ToLower(levelStr)
	switch normalizedLevel {
	case "error":
		return loggers.Error, nil
	case "warn":
		return loggers.Warn, nil
	case "info":
		return loggers.Info, nil
	case "debug":
		return loggers.Debug, nil
	default:
		return 0, fmt.Errorf("invalid log level: %s (must be one of: error, warn, info, debug)", levelStr)
	}
}

// Log represents the log primitive
type Log struct{}

var _ primitive_types.Primitive = &Log{}

// Name returns the name of the primitive
func (l *Log) Name() string {
	return "/gnd/log"
}

// Execute runs the log primitive
func (l *Log) Execute(args []interface{}) (interface{}, error) {

	if len(args) <= 1 {
		return nil, LogPrimitiveRequiresAtLeastTwoArguments
	}

	// Level
	levelStr, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("log: level must be a string, got %T", args[0])
	}

	// Convert level string to int
	level, err := ConvertLogLevel(levelStr)
	if err != nil {
		return nil, fmt.Errorf("log: level must be one of: error, warn, info, debug, got %s", levelStr)
	}

	// Convert remaining arguments to strings
	args = args[1:]

	str := ""
	if len(args) != 0 {
		for i, arg := range args {
			if i != 0 {
				str += " "
			}
			s, e := parsers.ParseString(arg)
			if e != nil {
				return nil, fmt.Errorf("log: failed to parse log message: %w", e)
			}
			str += s
		}
	}

	loggers.Printf(level, str)

	return str, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Log{})
}
