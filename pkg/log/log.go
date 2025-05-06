package log

import (
	"fmt"
	"os"
)

// Log levels
const (
	Error = iota
	Warn
	Info
	Debug
)

var Level = Error

// levelToString converts a log level to its string representation
func levelToString(level int) string {
	switch level {
	case Error:
		return "ERROR"
	case Warn:
		return "WARN"
	case Info:
		return "INFO"
	case Debug:
		return "DEBUG"
	default:
		return "UNKNOWN"
	}
}

// Printf logs a message at the specified level
func Printf(level int, format string, args ...interface{}) {
	if level <= Level {
		fmt.Fprintf(os.Stderr, "[%s]: %s\n", levelToString(level), fmt.Sprintf(format, args...))
	}
}
