package primitive

import "fmt"

type errorPrimitive struct{}

func (p *errorPrimitive) Name() string {
	return "/gnd/error"
}

func (p *errorPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("error expects 1 argument")
	}
	return nil, MakeError(args[0])
}

// MakeError creates an error with the given message
func MakeError(msg string) error {
	return fmt.Errorf(msg)
}

func init() {
	RegisterPrimitive(&errorPrimitive{})
}
