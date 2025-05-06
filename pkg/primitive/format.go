package primitive

import (
	"fmt"
)

type formatPrimitive struct{}

func (p *formatPrimitive) Name() string {
	return "/gnd/format"
}

func (p *formatPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) < 1 {
		return nil, fmt.Errorf("format expects at least 1 argument")
	}
	// Remove quotes from format string if present
	template := args[0]
	if len(template) >= 2 && template[0] == '"' && template[len(template)-1] == '"' {
		template = template[1 : len(template)-1]
	}
	// Convert remaining arguments to interface{}
	values := make([]interface{}, len(args)-1)
	for i := 1; i < len(args); i++ {
		value := args[i]
		// Try to parse as number first
		if num, err := ParseNumber(value); err == nil {
			values[i-1] = num
			continue
		}
		// If it's a quoted string, remove the quotes
		if len(value) >= 2 && value[0] == '"' && value[len(value)-1] == '"' {
			value = value[1 : len(value)-1]
		}
		// If it's a boolean string, convert to bool
		if value == "true" || value == "false" {
			values[i-1] = value == "true"
			continue
		}
		// Use as string
		values[i-1] = value
	}
	return Format(template, values...)
}

// Format fills placeholders in a template string
func Format(template string, args ...interface{}) (string, error) {
	// Handle empty template
	if template == "" {
		return "", nil
	}

	// Special case: single % is invalid but should return as is
	if template == "%" {
		return "%", nil
	}

	// Use fmt.Sprintf for the actual formatting
	// It will handle mismatched arguments by adding %!(EXTRA type=value) or %!(MISSING)
	return fmt.Sprintf(template, args...), nil
}

func init() {
	RegisterPrimitive(&formatPrimitive{})
}
