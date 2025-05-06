package primitive

import (
	"fmt"
	"strings"
)

type stringSplitPrimitive struct{}

func (p *stringSplitPrimitive) Name() string {
	return "/gnd/string-split"
}

func (p *stringSplitPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("string-split expects 2 arguments")
	}
	return StringSplit(args[0], args[1])
}

// StringSplit splits a string by a delimiter
func StringSplit(s, delim string) ([]string, error) {
	// Remove quotes if present
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		s = s[1 : len(s)-1]
	}
	if len(delim) >= 2 && delim[0] == '"' && delim[len(delim)-1] == '"' {
		delim = delim[1 : len(delim)-1]
	}

	// Split the string
	return strings.Split(s, delim), nil
}

func init() {
	RegisterPrimitive(&stringSplitPrimitive{})
}
