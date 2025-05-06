package primitive

import "fmt"

// Let is a primitive that binds a value to a name, or resets _ if no args
// let x y   => x = y
// let x     => x = _
// let       => _ = ""
type Let struct{}

func (l *Let) Name() string {
	return "/gnd/let"
}

func (l *Let) Execute(args []string) (interface{}, error) {
	switch len(args) {
	case 2:
		// let x y
		return args[1], nil
	case 1:
		// let x
		return args[0], nil
	case 0:
		// let
		return "", nil
	default:
		return nil, fmt.Errorf("let expects at most 2 arguments, got %d", len(args))
	}
}
