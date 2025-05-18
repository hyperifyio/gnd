package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// Predefined errors
var (
	AsyncErrNoArguments    = errors.New("async: requires a routine")
	AsyncErrInvalidRoutine = errors.New("async: routine must be an instruction array")
)

// Async represents the async primitive
type Async struct{}

var _ primitive_types.Primitive = &Async{}
var _ primitive_types.BlockSuccessResultHandler = &Async{}

// Name returns the name of the primitive
func (a *Async) Name() string {
	return "/gnd/async"
}

// Execute runs the async primitive
func (a *Async) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, AsyncErrNoArguments
	}

	// Get the routine
	var routine []*parsers.Instruction
	switch v := args[0].(type) {
	case []*parsers.Instruction:
		routine = v
	case *parsers.Instruction:
		routine = []*parsers.Instruction{v}
	default:
		return nil, AsyncErrInvalidRoutine
	}

	// Create a new task
	task := NewTask(routine, args[1:])

	return task, nil
}

func (a *Async) HandleBlockSuccessResult(result interface{}, interpreter primitive_types.Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error) {
	if task, ok := GetTask(result); ok {

		// Start the task in a goroutine
		go func() {
			// TODO: Execute the routine in the task's context
			// This will be implemented when we have access to the interpreter
			task.Mu.Lock()
			task.State = TaskStateCompleted
			task.Mu.Unlock()
		}()

		return task, nil
	}
	return result, nil
}

// HandleTaskResult processes a Task and returns the routine's output
func HandleTaskResult(i primitive_types.Interpreter, source string, task *Task) (interface{}, error) {
	i.LogDebug("[%s]: HandleTaskResult: executing routine with args: %v", source, task.Args)

	// Create a new interpreter for the routine
	interpreter := i.NewInterpreterWithParent(
		i.GetScriptDir(),
		map[string]interface{}{
			"_": task.Args,
		},
	)

	// Execute the routine
	result, err := interpreter.ExecuteInstructionBlock(source, task.Args, task.Routine)
	if err != nil {
		task.Mu.Lock()
		task.State = TaskStateError
		task.Error = err.Error()
		task.Mu.Unlock()
		return nil, err
	}

	// Update task state
	task.Mu.Lock()
	task.State = TaskStateCompleted
	task.Result = result
	task.Mu.Unlock()

	return result, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Async{})
}
