package core

import (
	"bufio"
	"fmt"
	"strings"
)

// ParseInstructionsString parses all instructions from a string
func ParseInstructionsString(content string, scriptDir string) ([]*Instruction, error) {
	var instructions []*Instruction
	scanner := bufio.NewScanner(strings.NewReader(content))
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		op, err := ParseInstruction(line, scriptDir)
		if err != nil {
			return nil, fmt.Errorf("error on line %d: %w", lineNum, err)
		}

		if op != nil {
			instructions = append(instructions, op)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return instructions, nil
}
