package parsers

import (
	"errors"
	"fmt"
	"strings"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

var DestinationMustBeAStringError = errors.New("destination must be a string literal")
var OpcodeMustBeAStringError = errors.New("opcode must be a string literal")
var EmptyInstructionError = errors.New("empty instruction")

// ParseInstruction parses a single GND instruction line
func ParseInstruction(source, line string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if len(line) == 0 || IsHashtag(line[0]) {
		loggers.Printf(loggers.Debug, "[%s]: Ignored line: %s", source, line)
		return nil, nil
	}

	tokens, err := TokenizeLine(line)
	if err != nil {
		return nil, fmt.Errorf("failed to tokenize line: %w", err)
	}
	if len(tokens) == 0 {
		return nil, EmptyInstructionError
	}

	// Parse optional destination
	var ok bool
	var dest *PropertyRef
	var opcode string
	dest, ok = GetPropertyRef(tokens[0])
	if !ok {
		dest = NewPropertyRef("_")
		opcode, ok = tokens[0].(string)
		if !ok {
			return nil, OpcodeMustBeAStringError
		}
		tokens = tokens[1:]
	} else {
		opcode, ok = tokens[1].(string)
		if !ok {
			return nil, OpcodeMustBeAStringError
		}
		tokens = tokens[2:]
	}

	// Collect remaining tokens as arguments
	var args []interface{}
	if len(tokens) > 0 {
		args = tokens
	} else {
		args = []interface{}{NewPropertyRef("_")}
	}

	loggers.Printf(loggers.Debug, "[%s]: Parsed line: %s: %v %v %v", source, line, opcode, dest, args)
	return &Instruction{
		Opcode:      opcode,
		Destination: dest,
		Arguments:   args,
	}, nil
}
