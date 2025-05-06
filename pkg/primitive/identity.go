package primitive

import "fmt"

// Identity is a primitive that returns its input unchanged
type Identity struct{}

func (i *Identity) Name() string {
	return "/gnd/identity"
}

func (i *Identity) Execute(args []string) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("identity expects 1 argument, got %d", len(args))
	}
	return args[0], nil
}
