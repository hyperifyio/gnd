package primitives

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

var (
	AsyncErrNoArguments    = errors.New("async: requires a routine")
	AsyncErrInvalidRoutine = errors.New("async: routine must be an instruction array")
)

type Async struct{}

var _ primitive_types.Primitive = &Async{}
var _ primitive_types.BlockSuccessResultHandler = &Async{}

// Name returns the name of the primitive
func (a *Async) Name() string { return "/gnd/async" }

// Execute runs the async primitive
func (a *Async) Execute(args []interface{}) (interface{}, error) {
	if len(args) == 0 {
		return nil, AsyncErrNoArguments
	}

	var routine []*parsers.Instruction
	switch v := args[0].(type) {
	case []*parsers.Instruction:
		routine = v
	case *parsers.Instruction:
		routine = []*parsers.Instruction{v}
	default:
		return nil, AsyncErrInvalidRoutine
	}

	return NewTask(routine, args[1:]), nil
}

// HandleBlockSuccessResult spawn the goroutine once the interpreter has the task in hand
func (a *Async) HandleBlockSuccessResult(
	result interface{},
	i primitive_types.Interpreter,
	_ *parsers.PropertyRef,
	_ []*parsers.Instruction,
) (interface{}, error) {

	var source = a.Name()

	task, ok := GetTask(result)
	if !ok {
		return result, nil // nothing to do
	}

	// Spawn a child interpreter with "_" initialised to the argument list.
	interp := i.NewInterpreterWithParent(
		i.GetScriptDir(),
		map[string]interface{}{
			"_": task.Args,
		},
	)

	// mark running, then launch worker
	task.SetState(TaskStateRunning)
	go func(childInterpreter primitive_types.Interpreter, name string, t *Task) {
		val, err := HandleTaskResult(childInterpreter, name, t)
		if err != nil {
			t.SetError(err)
		} else {
			t.SetCompleted(val)
		}
	}(interp, source, task)

	return task, nil
}

func init() { primitive_services.RegisterPrimitive(&Async{}) }

// HandleTaskResult runs the task's instruction block and returns the routine's output.
// It does NOT update task.State or write to task.done; the caller (the goroutine
// created in Async.HandleBlockSuccessResult) handles those concerns.
func HandleTaskResult(
	i primitive_types.Interpreter,
	source string,
	task *Task,
) (interface{}, error) {

	i.LogDebug("[%s] HandleTaskResult: args=%v", source, task.Args)

	// Run the instruction array and return its value or error exactly as is.
	return i.ExecuteInstructionBlock(source, task.Args, task.Routine)
}
