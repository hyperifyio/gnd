package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive"
)

// Log levels
const (
	LogError = iota
	LogInfo
	LogDebug
)

var logLevel = LogError

func logf(level int, format string, args ...interface{}) {
	if level <= logLevel {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}
}

// Default opcode mapping
var defaultOpcodeMap = map[string]string{
	"prompt": "/gnd/prompt",
	"let":    "/gnd/let",
}

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
	// Map to full path if in defaultOpcodeMap
	if mapped, ok := defaultOpcodeMap[opcode]; ok {
		opcode = mapped
	}

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
func (i *Interpreter) ExecuteInstruction(op *Instruction, idx int) error {
	if op == nil {
		return nil
	}

	logf(LogDebug, "[DEBUG] Executing instruction %d: %s %s %v", idx+1, op.Opcode, op.Destination, op.Arguments)

	var resolvedArgs []string
	if op.Opcode == "/gnd/let" && len(op.Arguments) == 0 && op.Destination != "_" {
		// let args  => bind _ to args
		if val, ok := i.Slots["_"]; ok {
			resolvedArgs = []string{fmt.Sprintf("%v", val)}
		} else {
			resolvedArgs = []string{""}
		}
	} else {
		resolvedArgs = make([]string, len(op.Arguments))
		for j, arg := range op.Arguments {
			if val, ok := i.Slots[arg]; ok {
				resolvedArgs[j] = fmt.Sprintf("%v", val)
			} else if op.Opcode == "/gnd/let" && len(op.Arguments) == 1 {
				// let x y (already handled above, but keep for completeness)
				if val, ok := i.Slots["_"]; ok {
					resolvedArgs[j] = fmt.Sprintf("%v", val)
				} else {
					resolvedArgs[j] = ""
				}
			} else {
				resolvedArgs[j] = arg
			}
		}
	}
	logf(LogDebug, "[DEBUG] Resolved args: %v", resolvedArgs)

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		logf(LogError, "[ERROR] Unknown opcode: %s", op.Opcode)
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		logf(LogError, "[ERROR] Primitive error: %v", err)
		return err
	}

	logf(LogDebug, "[DEBUG] Result: %v", result)
	i.Slots[op.Destination] = result
	logf(LogDebug, "[DEBUG] Slots: %+v", i.Slots)
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

func parseLogLevel(level string) int {
	switch strings.ToLower(level) {
	case "debug":
		return LogDebug
	case "info":
		return LogInfo
	default:
		return LogError
	}
}

func main() {
	logLevelFlag := flag.String("log-level", "error", "Set log level: debug, info, error")
	interactiveFlag := flag.Bool("interactive", false, "Run in interactive mode")
	inputFlag := flag.String("input", "", "Input for the script")
	flag.Parse()

	logLevel = parseLogLevel(*logLevelFlag)

	args := flag.Args()
	if len(args) < 1 {
		fmt.Println("Usage: gnd [--log-level debug|info|error] [--interactive] [--input <text>] <file-path>")
		os.Exit(1)
	}

	contentBytes, err := os.ReadFile(args[0])
	if err != nil {
		logf(LogError, "[ERROR] Error reading file: %v", err)
		os.Exit(1)
	}
	content := string(contentBytes)

	instructions, err := ParseFile(content)
	if err != nil {
		logf(LogError, "[ERROR] Error parsing file: %v", err)
		os.Exit(1)
	}

	interpreter := &Interpreter{Slots: make(map[string]interface{})}

	// Handle interactive mode
	if *interactiveFlag {
		if *inputFlag != "" {
			logf(LogError, "[ERROR] Cannot use --input with --interactive mode")
			os.Exit(1)
		}
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			logf(LogError, "[ERROR] Error reading input: %v", err)
			os.Exit(1)
		}
		interpreter.Slots["_"] = strings.TrimSpace(input)
	} else {
		// Handle non-interactive mode
		if *inputFlag != "" {
			interpreter.Slots["_"] = *inputFlag
		} else {
			// Read from stdin if not a terminal
			stdinStat, _ := os.Stdin.Stat()
			if (stdinStat.Mode() & os.ModeCharDevice) == 0 {
				stdinBytes, _ := io.ReadAll(os.Stdin)
				interpreter.Slots["_"] = strings.TrimRight(string(stdinBytes), "\n")
			} else {
				logf(LogError, "[ERROR] Input is required. Either provide it through stdin or use --input parameter")
				os.Exit(1)
			}
		}
	}

	var lastResult interface{}

	for i, op := range instructions {
		err = interpreter.ExecuteInstruction(op, i)
		if err != nil {
			logf(LogError, "[ERROR] Error executing instruction %d: %v", i+1, err)
			os.Exit(1)
		}
		if op != nil {
			lastResult = interpreter.Slots[op.Destination]
		}
	}

	if lastResult != nil {
		fmt.Println(lastResult)
	}
}
