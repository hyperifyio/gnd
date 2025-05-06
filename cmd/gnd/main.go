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
	"prompt":    "/gnd/prompt",
	"let":       "/gnd/let",
	"select":    "/gnd/select",
	"concat":    "/gnd/concat",
	"lowercase": "/gnd/lowercase",
	"uppercase": "/gnd/uppercase",
	"trim":      "/gnd/trim",
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

	var tokens []string
	var current strings.Builder
	inString := false
	escape := false

	for i := 0; i < len(line); i++ {
		c := line[i]

		if escape {
			if inString {
				// Only process escape sequences inside strings
				switch c {
				case 'n':
					current.WriteByte('\n')
				case 't':
					current.WriteByte('\t')
				case 'r':
					current.WriteByte('\r')
				case '\\':
					current.WriteByte('\\')
				case '"':
					current.WriteByte('"')
				default:
					current.WriteByte('\\')
					current.WriteByte(c)
				}
			} else {
				current.WriteByte('\\')
				current.WriteByte(c)
			}
			escape = false
			continue
		}

		if c == '\\' {
			escape = true
			continue
		}

		if c == '"' {
			if inString {
				// End of string
				tokens = append(tokens, current.String())
				current.Reset()
				inString = false
			} else {
				// Start of string
				if current.Len() > 0 {
					tokens = append(tokens, current.String())
					current.Reset()
				}
				inString = true
			}
			continue
		}

		if !inString && (c == ' ' || c == '\t') {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			continue
		}

		current.WriteByte(c)
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

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

// unescapeString converts escape sequences in a string literal to their actual values
func unescapeString(s string) string {
	var result strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			switch s[i+1] {
			case 'n':
				result.WriteByte('\n')
			case 't':
				result.WriteByte('\t')
			case 'r':
				result.WriteByte('\r')
			case '\\':
				result.WriteByte('\\')
			case '"':
				result.WriteByte('"')
			default:
				result.WriteByte(s[i])
				result.WriteByte(s[i+1])
			}
			i++ // Skip the next character
		} else {
			result.WriteByte(s[i])
		}
	}
	return result.String()
}

// ExecuteInstruction executes a single GND instruction
func (i *Interpreter) ExecuteInstruction(op *Instruction, idx int) error {
	if op == nil {
		return nil
	}

	logf(LogDebug, "[DEBUG] %s %s %v", op.Opcode, op.Destination, op.Arguments)

	// Resolve arguments
	resolvedArgs := make([]interface{}, len(op.Arguments))
	for j, arg := range op.Arguments {
		// If the argument is a string literal (starts and ends with quotes)
		if len(arg) >= 2 && arg[0] == '"' && arg[len(arg)-1] == '"' {
			// Remove the quotes and unescape
			unescaped := unescapeString(arg[1 : len(arg)-1])
			resolvedArgs[j] = unescaped
		} else if val, ok := i.Slots[arg]; ok {
			// If it's a defined variable, use its value
			resolvedArgs[j] = val
		} else {
			// Otherwise use the literal value
			resolvedArgs[j] = arg
		}
	}

	// If no arguments provided, use current value of _
	if len(resolvedArgs) == 0 {
		if val, ok := i.Slots["_"]; ok {
			resolvedArgs = []interface{}{val}
		} else {
			resolvedArgs = []interface{}{""}
		}
	}

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		return fmt.Errorf("%s: %v", op.Opcode, err)
	}

	// For debug output, escape newlines to make them visible
	debugResult := fmt.Sprintf("%v", result)
	debugResult = strings.ReplaceAll(debugResult, "\n", "\\n")
	logf(LogDebug, "[DEBUG] %s = %s", op.Destination, debugResult)

	// Store the result in the destination slot
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
