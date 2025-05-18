package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"time"
)

// Predefined errors
var (
	WaitErrNoArguments     = errors.New("wait: requires an argument")
	WaitErrInvalidArgument = errors.New("wait: argument must be a task or number")
)

// Wait represents the wait primitive
type Wait struct{}

var _ primitive_types.Primitive = &Wait{}

// Name returns the name of the primitive
func (w *Wait) Name() string {
	return "/gnd/wait"
}

// Execute runs the wait primitive
func (w *Wait) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, WaitErrNoArguments
	}

	// Check if the argument is a number (duration in milliseconds)
	if duration, ok := args[0].(float64); ok {
		time.Sleep(time.Duration(duration) * time.Millisecond)
		return true, nil
	}

	// Check if the argument is a task
	task, ok := args[0].(*Task)
	if !ok {
		return nil, WaitErrInvalidArgument
	}

	// Wait for the task to complete
	for {
		task.Mu.Lock()
		state := task.State
		task.Mu.Unlock()

		if state != TaskStatePending {
			// Return [flag value] list
			if state == TaskStateCompleted {
				return []interface{}{true, task.Result}, nil
			}
			return []interface{}{false, task.Error}, nil
		}

		time.Sleep(10 * time.Millisecond)
	}
}

func init() {
	primitive_services.RegisterPrimitive(&Wait{})
}
