package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Predefined errors
var (
	StatusErrNoArguments     = errors.New("status: requires a task")
	StatusErrInvalidArgument = errors.New("status: argument must be a task")
)

// Status represents the status primitive
type Status struct{}

var _ primitive_types.Primitive = &Status{}

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
	primitive_services.RegisterPrimitive(&Status{})
}
