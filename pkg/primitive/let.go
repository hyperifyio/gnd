package primitive

// Let is a primitive that binds a value to a name
type Let struct{}

func (l *Let) Name() string {
	return "/gnd/let"
}

func (l *Let) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return "", nil
	}
	return args[0], nil
}
