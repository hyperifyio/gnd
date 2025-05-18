package primitives

import (
	"errors"
	"time"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Predefined errors
var (
	WaitErrNoArguments     = errors.New("wait: requires an argument")
	WaitErrInvalidArgument = errors.New("wait: argument must be a task or number")
	WaitErrTooManyArgs     = errors.New("wait: too many arguments")
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

	if len(args) > 1 {
		return nil, WaitErrTooManyArgs
	}

	// Check if the argument is a number (duration in milliseconds)
	switch v := args[0].(type) {
	case float64:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case int:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case int64:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case int32:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case int16:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case int8:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case uint:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case uint64:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case uint32:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case uint16:
		time.Sleep(time.Duration(v) * time.Millisecond)
	case uint8:
		time.Sleep(time.Duration(v) * time.Millisecond)
	default:
		// Check if the argument is a task
		task, ok := args[0].(*Task)
		if !ok {
			return nil, WaitErrInvalidArgument
		}

		// Wait for the task to complete and return its result
		result, err := task.Await()
		if err != nil {
			return []interface{}{false, err.Error()}, nil
		}
		return []interface{}{true, result}, nil
	}
	return true, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Wait{})
}
