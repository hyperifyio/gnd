package log

import (
	"fmt"
	"strings"
)

// StringifyValue formats a value for logging by escaping newlines
func StringifyValue(value interface{}) string {
	debugResult := fmt.Sprintf("%v", value)
	return strings.ReplaceAll(debugResult, "\n", "\\n")
}
