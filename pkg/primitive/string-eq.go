package primitive

import "fmt"

type stringEqPrimitive struct{}

func (p *stringEqPrimitive) Name() string {
	return "/gnd/string-eq"
}

func (p *stringEqPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 2 {
		return nil, fmt.Errorf("string-eq expects 2 arguments")
	}
	return StringEq(args[0], args[1])
}

// StringEq checks if two strings are equal
func StringEq(s1 string, s2 string) (bool, error) {
	return s1 == s2, nil
}

func init() {
	RegisterPrimitive(&stringEqPrimitive{})
}
