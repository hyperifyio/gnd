package primitive

import (
	"encoding/json"
	"fmt"
)

type selectPrimitive struct{}

func (p *selectPrimitive) Name() string {
	return "/gnd/select"
}

func (p *selectPrimitive) Execute(args []string) (interface{}, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("select expects 3 arguments")
	}
	return SelectValue(args[0], args[1], args[2])
}

// SelectValue chooses between two values based on a condition
func SelectValue(cond, then, elseVal string) (interface{}, error) {
	// Parse the condition
	var b bool
	if err := json.Unmarshal([]byte(cond), &b); err != nil {
		return nil, fmt.Errorf("invalid condition: %v", err)
	}

	// Return the appropriate value
	if b {
		var result interface{}
		if err := json.Unmarshal([]byte(then), &result); err != nil {
			return nil, fmt.Errorf("invalid then value: %v", err)
		}
		return result, nil
	}

	var result interface{}
	if err := json.Unmarshal([]byte(elseVal), &result); err != nil {
		return nil, fmt.Errorf("invalid else value: %v", err)
	}
	return result, nil
}

func init() {
	RegisterPrimitive(&selectPrimitive{})
}
