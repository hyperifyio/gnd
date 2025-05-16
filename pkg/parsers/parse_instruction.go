package parsers

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/log"
	"strings"
)

// ParseInstruction parses a single GND instruction line
func ParseInstruction(source, line string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		log.Printf(log.Debug, "[%s]: Ignored line: %s", source, line)
		return nil, nil
	}

	tokens, err := TokenizeLine(line)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize line: %w", err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty instruction")
	}

	// Type assert opcode to string
	opcode, ok := tokens[0].(string)
	if !ok {
		return nil, fmt.Errorf("opcode must be a string")
	}

	var dest string
	var args []interface{}
	if len(tokens) > 1 {

		// Type assert destination to string
		dest, ok = tokens[1].(string)
		if !ok {
			return nil, fmt.Errorf("destination must be a string")
		}

		// Collect remaining tokens as arguments
		if len(tokens) > 2 {
			args = tokens[2:]
		} else {
			args = []interface{}{&PropertyRef{"_"}}
		}

	} else {
		dest = "_"
		args = []interface{}{&PropertyRef{"_"}}
	}

	// Check if this is a subroutine call
	log.Printf(log.Debug, "[%s]: Parsed line: %s: %v %v %v", source, line, opcode, dest, args)
	return &Instruction{
		Opcode:      opcode,
		Destination: dest,
		Arguments:   args,
	}, nil

}
