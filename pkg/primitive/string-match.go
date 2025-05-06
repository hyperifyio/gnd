package primitive

import (
	"fmt"
	"regexp"
)

type stringMatchPrimitive struct{}

func (p *stringMatchPrimitive) Name() string {
	return "/gnd/string-match"
}

func (p *stringMatchPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("string-match expects 2 arguments")
	}
	return StringMatch(args[0], args[1])
}

// StringMatch checks if a string matches a regex pattern
func StringMatch(s, pattern string) (bool, error) {
	// Remove quotes if present
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		s = s[1 : len(s)-1]
	}
	if len(pattern) >= 2 && pattern[0] == '"' && pattern[len(pattern)-1] == '"' {
		pattern = pattern[1 : len(pattern)-1]
	}

	// Compile regex with timeout
	re, err := regexp.Compile(pattern)
	if err != nil {
		return false, fmt.Errorf("invalid regex pattern: %v", err)
	}

	// Find all matches
	matches := re.FindAllString(s, -1)
	return len(matches) > 0, nil
}

func init() {
	RegisterPrimitive(&stringMatchPrimitive{})
}
