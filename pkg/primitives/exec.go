package primitives

import (
	"errors"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	primitive_types2 "github.com/hyperifyio/gnd/pkg/primitive_types"
)

var ExecRequiresRoutineError = errors.New("exec: requires a routine")
var ExecRoutineInvalidError = errors.New("exec: routine must be an instruction array or instruction")
var ExecRoutineExecuteFailedError = errors.New("exec: routine execution failed")

// Exec represents the exec primitive
type Exec struct {
}

var _ primitive_types2.BlockSuccessResultHandler = &Exec{}

// Name returns the name of the primitive
func (c *Exec) Name() string {
	return "/gnd/exec"
}

// Execute runs the exec primitive
func (c *Exec) Execute(args []interface{}) (interface{}, error) {

	var routine []*parsers.Instruction
	var routineArgs []interface{}

	if len(args) == 0 {
		return nil, ExecRequiresRoutineError
	}

	// Get the routine
	switch v := args[0].(type) {
	case []*parsers.Instruction:
		routine = v
		routineArgs = args[1:]
	case *parsers.Instruction:
		routine = []*parsers.Instruction{v}
		routineArgs = args[1:]
	default:
		loggers.Printf(loggers.Error, "exec: routine must be an instruction array, got %T", args[0])
		return nil, ExecRoutineInvalidError
	}

	// Return an ExecResult for the interpreter to handle
	return NewExecResult(routine, routineArgs), nil
}

// HandleBlockSuccessResult handles exec results
func (c *Exec) HandleBlockSuccessResult(result interface{}, i primitive_types2.Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error) {
	if execResult, ok := GetExecResult(result); ok {
		i.LogDebug("[/gnd/exec]: ExecutePrimitive: exec result detected: %v", execResult)
		res, err := HandleExecResult(i, execResult)
		if err != nil {
			return nil, err
		}
		i.SetSlot(destination.Name, res)
		return res, nil
	}
	return result, nil
}

// HandleExecResult processes an ExecResult and returns the routine's output
func HandleExecResult(i primitive_types2.Interpreter, execResult *ExecResult) (interface{}, error) {

	var args = execResult.Args
	var routine = execResult.Routine

	i.LogDebug("[/gnd/exec]: HandleExecResult: executing routine with args: %v", args)

	// Create a new interpreter for the routine
	interpreter := i.NewInterpreterWithParent(
		i.GetScriptDir(),
		map[string]interface{}{
			"_": args,
		},
	)

	// Execute the routine
	result, err := interpreter.ExecuteInstructionBlock("/gnd/exec", args, routine)
	if err != nil {
		i.LogError("[/gnd/exec]: HandleExecResult: routine execution failed: %v", err)
		return nil, ExecRoutineExecuteFailedError
	}

	return result, nil
}

func init() {
	primitive_services.RegisterPrimitive(&Exec{})
}
