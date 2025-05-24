// Package logging provides logging functionality for the BitNet project.
// It includes debug logging and other logging utilities.
package logging

import "github.com/hyperifyio/gnd/pkg/loggers"

// DebugLogf logs debug information using the configured logger.
// It formats the message according to the format specifier and arguments.
func DebugLogf(format string, args ...interface{}) {
	loggers.Printf(loggers.Debug, format, args...)
}
