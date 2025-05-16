package parsers

import (
	"bufio"
	"fmt"
	"github.com/hyperifyio/gnd/pkg/log"
	"strings"
)

// ParseInstructionLines parses all instructions from a string
func ParseInstructionLines(source, content string) ([]*Instruction, error) {
	var instructions []*Instruction
	scanner := bufio.NewScanner(strings.NewReader(content))
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		log.Printf(log.Debug, "[%s]: Parsed line %d: %s", source, lineNum, line)

		op, err := ParseInstruction(fmt.Sprintf("%s:%d", source, lineNum), line)
		if err != nil {
			return nil, fmt.Errorf("[%s]: error on line %d: %w", source, lineNum, err)
		}

		if op != nil {
			instructions = append(instructions, op)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("[%s]: error reading file: %w", source, err)
	}

	return instructions, nil
}
