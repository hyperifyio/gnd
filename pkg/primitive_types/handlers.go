package primitive_types

import (
	"github.com/hyperifyio/gnd/pkg/parsers"
)

// BlockSuccessResultHandler handles successful block execution results
type BlockSuccessResultHandler interface {
	HandleBlockSuccessResult(result interface{}, interpreter Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error)
}

// BlockErrorResultHandler handles block execution errors
type BlockErrorResultHandler interface {
	HandleBlockErrorResult(err error, interpreter Interpreter, destination *parsers.PropertyRef, block []*parsers.Instruction) (interface{}, error)
}
