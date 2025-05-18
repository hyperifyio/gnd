package primitives

import (
	"errors"
)

// Predefined errors
var (
	StatusErrNoArguments     = errors.New("status: requires a task")
	StatusErrInvalidArgument = errors.New("status: argument must be a task")
)

// Status represents the status primitive
type Status struct{}

// Name returns the name of the primitive
func (s *Status) Name() string {
	return "/gnd/status"
}

// Execute runs the status primitive
func (s *Status) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, StatusErrNoArguments
	}

	// Check if the argument is a task
	task, ok := args[0].(*Task)
	if !ok {
		return nil, StatusErrInvalidArgument
	}

	// Return the current state
	task.Mu.Lock()
	state := task.State
	task.Mu.Unlock()

	return string(state), nil
}

func init() {
	RegisterPrimitive(&Status{})
}
