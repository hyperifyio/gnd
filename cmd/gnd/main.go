package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive"
)

// Instruction represents a parsed GND instruction
type Instruction struct {
	Opcode      string
	Destination string
	Arguments   []string
}

// Interpreter represents the execution environment
type Interpreter struct {
	Slots map[string]interface{}
}

// ParseInstruction parses a single GND instruction
func ParseInstruction(line string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil
	}

	tokens := strings.Fields(line)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty instruction")
	}

	opcode := tokens[0]
	if !strings.HasPrefix(opcode, "/gnd/") {
		return nil, fmt.Errorf("invalid opcode prefix: %s (must start with /gnd/)", opcode)
	}

	// Parse destination
	var dest string
	var args []string
	if len(tokens) > 1 {
		dest = tokens[1]
		args = tokens[2:]
	} else {
		dest = "_"
	}

	return &Instruction{
		Opcode:      opcode,
		Destination: dest,
		Arguments:   args,
	}, nil
}

// ExecuteInstruction executes a single GND instruction
func (i *Interpreter) ExecuteInstruction(op *Instruction) error {
	if op == nil {
		return nil
	}

	resolvedArgs := make([]string, len(op.Arguments))
	for j, arg := range op.Arguments {
		if val, ok := i.Slots[arg]; ok {
			resolvedArgs[j] = fmt.Sprintf("%v", val)
		} else {
			resolvedArgs[j] = arg
		}
	}

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		return err
	}

	i.Slots[op.Destination] = result
	return nil
}

// ParseFile parses all instructions from a file
func ParseFile(content string) ([]*Instruction, error) {
	var instructions []*Instruction
	scanner := bufio.NewScanner(strings.NewReader(content))
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		op, err := ParseInstruction(line)
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

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: gnd <file-path>")
		os.Exit(1)
	}

	contentBytes, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}
	content := string(contentBytes)

	instructions, err := ParseFile(content)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing file: %v\n", err)
		os.Exit(1)
	}

	interpreter := &Interpreter{Slots: make(map[string]interface{})}

	for i, op := range instructions {
		err = interpreter.ExecuteInstruction(op)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error executing instruction %d: %v\n", i+1, err)
			os.Exit(1)
		}
	}
}
