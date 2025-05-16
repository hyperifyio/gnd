package primitive

import (
	"fmt"
	"os"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// Print represents the print primitive
type Print struct{}

// Name returns the name of the primitive
func (p *Print) Name() string {
	return "/gnd/print"
}

// Execute runs the print primitive
func (p *Print) Execute(args []interface{}) (interface{}, error) {
	output, err := parsers.ParseString(args)
	if err != nil {
		return nil, fmt.Errorf("print: %v", err)
	}
	fmt.Fprint(os.Stdout, output)
	return output, nil
}

func init() {
	RegisterPrimitive(&Print{})
}
