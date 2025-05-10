package core

import (
	"fmt"
	"strings"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// ParseInstruction parses a single GND instruction
func ParseInstruction(line string, scriptDir string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil
	}

	tokens, err := parsers.TokenizeLine(line)
	if err != nil {
		return nil, err
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
		args = tokens[2:]
	} else {
		dest = "_"
	}

	// Check if this is a subroutine call
	isSubroutine, subroutinePath := ResolveSubroutinePath(opcode, scriptDir)
	if !isSubroutine {
		if mapped, ok := DefaultOpcodeMap[opcode]; ok {
			// Only map to full path if it's not a subroutine
			opcode = mapped
		}
	}

	return &Instruction{
		Opcode:         opcode,
		Destination:    dest,
		Arguments:      args,
		IsSubroutine:   isSubroutine,
		SubroutinePath: subroutinePath,
	}, nil
}
