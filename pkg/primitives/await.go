package primitives

import (
	"errors"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

func init() {
	primitive_services.RegisterPrimitive(&Await{})
}

var (
	AwaitErrMissingTask    = errors.New("await: missing task argument")
	AwaitErrInvalidOperand = errors.New("await: operand is not a task")
	AwaitErrTooManyArgs    = errors.New("await: expects 1 argument")
)

type Await struct{}

var _ primitive_types.Primitive = &Await{}

func (a *Await) Name() string { return "/gnd/await" }

func (a *Await) Execute(args []interface{}) (interface{}, error) {

	var operand interface{}
	switch len(args) {
	case 0:
		// interpreter should have placed "_" in args already; defensive fallback
		return nil, AwaitErrMissingTask
	case 1:
		operand = args[0]
	default:
		return nil, AwaitErrTooManyArgs
	}

	task, ok := GetTask(operand)
	if !ok {
		return nil, AwaitErrInvalidOperand
	}

	res, err := task.Await()
	if err != nil {
		// Re-throw inside VM semantics
		return nil, err
	}
	return res, nil
}
