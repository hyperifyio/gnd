package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/hyperifyio/gnd/pkg/log"
	"github.com/hyperifyio/gnd/pkg/primitive"
	"github.com/hyperifyio/gnd/pkg/units"
)

// Default opcode mapping
var defaultOpcodeMap = map[string]string{
	"prompt":    "/gnd/prompt",
	"let":       "/gnd/let",
	"select":    "/gnd/select",
	"concat":    "/gnd/concat",
	"lowercase": "/gnd/lowercase",
	"uppercase": "/gnd/uppercase",
	"trim":      "/gnd/trim",
	"print":     "/gnd/print",
	"log":       "/gnd/log",
	"error":     "/gnd/error",
	"warn":      "/gnd/warn",
	"info":      "/gnd/info",
	"debug":     "/gnd/debug",
	"exit":      "/gnd/exit",
	"return":    "/gnd/return",
}

// Instruction represents a parsed GND instruction
type Instruction struct {
	Opcode         string
	Destination    string
	Arguments      []string
	IsSubroutine   bool
	SubroutinePath string
}

// Interpreter represents the execution environment
type Interpreter struct {
	Slots       map[string]interface{}
	Subroutines map[string][]*Instruction
	ScriptDir   string // Directory of the currently executing script
	LogIndent   int    // Current log indentation level
	UnitsFS     fs.FS  // Embedded filesystem containing GND units
}

// NewInterpreter creates a new interpreter instance
func NewInterpreter(scriptDir string) *Interpreter {
	return &Interpreter{
		Slots:       make(map[string]interface{}),
		Subroutines: make(map[string][]*Instruction),
		ScriptDir:   scriptDir,
		LogIndent:   0,
		UnitsFS:     units.GetUnitsFS(),
	}
}

// getLogPrefix returns the current log prefix based on indentation
func (i *Interpreter) getLogPrefix() string {
	if i.LogIndent == 0 {
		return ""
	}
	return strings.Repeat("  ", i.LogIndent)
}

// logDebug logs a debug message with proper indentation
func (i *Interpreter) logDebug(format string, args ...interface{}) {
	prefix := i.getLogPrefix()
	log.Printf(log.Debug, prefix+format, args...)
}

// LoadSubroutine loads a subroutine from a file
func (i *Interpreter) LoadSubroutine(name string) error {
	// Check if subroutine is already loaded
	if _, ok := i.Subroutines[name]; ok {
		return nil
	}

	// First try to find the subroutine in the script's directory
	subroutinePath := filepath.Join(i.ScriptDir, name+".gnd")
	contentBytes, err := os.ReadFile(subroutinePath)
	if err != nil {
		// If not found in script directory, try embedded units
		contentBytes, err = fs.ReadFile(i.UnitsFS, name+".gnd")
		if err != nil {
			return fmt.Errorf("subroutine not found: %s", name)
		}
	}

	// Parse the subroutine file
	instructions, err := ParseFile(string(contentBytes), i.ScriptDir)
	if err != nil {
		return fmt.Errorf("error parsing subroutine %s: %w", name, err)
	}

	// Store the subroutine
	i.Subroutines[name] = instructions
	return nil
}

// ExecuteSubroutine executes a subroutine with its own context
func (i *Interpreter) ExecuteSubroutine(name string) error {
	// Load the subroutine if not already loaded
	if err := i.LoadSubroutine(name); err != nil {
		return err
	}

	// Create a new interpreter for the subroutine with increased indentation
	subInterpreter := NewInterpreter(i.ScriptDir)
	subInterpreter.LogIndent = i.LogIndent + 1

	// Copy the current value of _ to the subroutine's args
	if val, ok := i.Slots["_"]; ok {
		i.logDebug("Copying value %v from main _ to subroutine args and _", val)
		subInterpreter.Slots["args"] = val
		subInterpreter.Slots["_"] = val
	} else {
		i.logDebug("No value in main _ to copy to subroutine")
	}

	// Log subroutine entry with current value
	i.logDebug("[ENTER SUBROUTINE] %s with input: %v", name, subInterpreter.Slots["args"])

	// Execute each instruction in the subroutine
	for _, op := range i.Subroutines[name] {
		err := subInterpreter.ExecuteInstruction(op, 0)
		if err != nil {
			return fmt.Errorf("error in subroutine %s: %w", name, err)
		}
	}

	// Copy the result back to the main interpreter's _
	if val, ok := subInterpreter.Slots["_"]; ok {
		i.logDebug("Copying value %v from subroutine _ back to main _", val)
		i.Slots["_"] = val
	} else {
		i.logDebug("No value in subroutine _ to copy back to main")
	}

	// Log subroutine exit with result
	i.logDebug("[EXIT SUBROUTINE] %s with output: %v", name, i.Slots["_"])

	return nil
}

// ExitError is a special error type that signals a normal program exit
type ExitError struct {
	code int
}

func (e *ExitError) Error() string {
	return fmt.Sprintf("exit with code %d", e.code)
}

// ParseInstruction parses a single GND instruction
func ParseInstruction(line string, scriptDir string) (*Instruction, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil
	}

	var tokens []string
	var current strings.Builder
	inString := false
	escape := false
	inArray := false

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

		if c == '[' && !inString {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			current.WriteByte('[')
			inArray = true
			continue
		}

		if c == ']' && !inString && inArray {
			current.WriteByte(']')
			tokens = append(tokens, current.String())
			current.Reset()
			inArray = false
			continue
		}

		if !inString && !inArray && (c == ' ' || c == '\t') {
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
	var dest string
	var args []string
	if len(tokens) > 1 {
		dest = tokens[1]
		args = tokens[2:]
	} else {
		dest = "_"
	}

	// For return primitive, join unquoted strings with spaces
	if opcode == "return" {
		// Check if all arguments are unquoted strings
		allUnquoted := true
		for _, arg := range args {
			if len(arg) >= 2 && arg[0] == '"' && arg[len(arg)-1] == '"' {
				allUnquoted = false
				break
			}
		}
		if allUnquoted && len(args) > 0 {
			args = []string{strings.Join(args, " ")}
		}
	}

	// Check if this is a subroutine call
	isSubroutine := false
	subroutinePath := ""
	if _, err := os.Stat(filepath.Join(scriptDir, opcode+".gnd")); err == nil {
		isSubroutine = true
		subroutinePath = opcode + ".gnd"
		// For subroutines, we keep the original opcode
	} else if _, err := fs.Stat(units.GetUnitsFS(), opcode+".gnd"); err == nil {
		isSubroutine = true
		subroutinePath = opcode + ".gnd"
		// For subroutines, we keep the original opcode
	} else if mapped, ok := defaultOpcodeMap[opcode]; ok {
		// Only map to full path if it's not a subroutine
		opcode = mapped
	}

	return &Instruction{
		Opcode:         opcode,
		Destination:    dest,
		Arguments:      args,
		IsSubroutine:   isSubroutine,
		SubroutinePath: subroutinePath,
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

	// Handle subroutine calls
	if op.IsSubroutine {
		// Extract the base name without .gnd extension
		baseName := strings.TrimSuffix(op.SubroutinePath, ".gnd")
		i.logDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

		// Resolve arguments
		resolvedArgs := make([]interface{}, len(op.Arguments))
		for j, arg := range op.Arguments {
			// If the argument is a string literal (starts and ends with quotes)
			if len(arg) >= 2 && arg[0] == '"' && arg[len(arg)-1] == '"' {
				// Remove the quotes and unescape
				unescaped := unescapeString(arg[1 : len(arg)-1])
				resolvedArgs[j] = unescaped
			} else if arg == "[]" {
				// Special case for empty array
				resolvedArgs[j] = []interface{}{}
			} else if val, ok := i.Slots[arg]; ok {
				// If it's a defined variable, use its value
				resolvedArgs[j] = val
			} else {
				// Otherwise use the literal value
				resolvedArgs[j] = arg
			}
		}

		// Store the resolved arguments in _
		i.logDebug("Storing resolved args %v in _ before subroutine call", resolvedArgs)
		i.Slots["_"] = resolvedArgs

		// Execute the subroutine
		err := i.ExecuteSubroutine(baseName)
		if err != nil {
			return err
		}

		// Store the result in the destination slot
		if val, ok := i.Slots["_"]; ok {
			i.logDebug("Storing subroutine result %v in destination %s", val, op.Destination)
			i.Slots[op.Destination] = val
		} else {
			i.logDebug("No result in _ to store in destination %s", op.Destination)
		}

		return nil
	}

	// Log regular instruction
	i.logDebug("%s %s %v", op.Opcode, op.Destination, op.Arguments)

	// Resolve arguments
	resolvedArgs := make([]interface{}, len(op.Arguments))
	for j, arg := range op.Arguments {
		// If the argument is a string literal (starts and ends with quotes)
		if len(arg) >= 2 && arg[0] == '"' && arg[len(arg)-1] == '"' {
			// Remove the quotes and unescape
			unescaped := unescapeString(arg[1 : len(arg)-1])
			resolvedArgs[j] = unescaped
		} else if arg == "[]" {
			// Special case for empty array
			resolvedArgs[j] = []interface{}{}
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
			i.logDebug("Using current value of _ (%v) as argument", val)
			resolvedArgs = []interface{}{val}
		} else {
			i.logDebug("No value in _, using empty string as argument")
			resolvedArgs = []interface{}{""}
		}
	}

	// For return primitive, handle unquoted strings
	if op.Opcode == "/gnd/return" {
		// Check if all arguments are unquoted strings
		allUnquoted := true
		for _, arg := range op.Arguments {
			if len(arg) >= 2 && arg[0] == '"' && arg[len(arg)-1] == '"' {
				allUnquoted = false
				break
			}
		}
		if allUnquoted && len(op.Arguments) > 0 {
			i.logDebug("Joining unquoted strings with spaces: %v", op.Arguments)
			resolvedArgs = []interface{}{strings.Join(op.Arguments, " ")}
		}
	}

	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	// Set destination and subroutine flag for return primitive
	if op.Opcode == "/gnd/return" {
		if ret, ok := prim.(*primitive.Return); ok {
			ret.IsSubroutine = op.IsSubroutine
			ret.Destination = op.Destination
		}
	}

	result, err := prim.Execute(resolvedArgs)
	if err != nil {
		return fmt.Errorf("%s: %v", op.Opcode, err)
	}

	i.logDebug("primitive result: %v", result)

	// Check if this is a return with exit signal
	if resultMap, ok := result.(map[string]interface{}); ok {
		i.logDebug("result is a map: %v", resultMap)
		if exit, ok := resultMap["exit"].(bool); ok && exit {
			i.logDebug("exit signal detected")
			// Get the exit code if provided
			exitCode := 0
			if code, ok := resultMap["code"].(int); ok {
				exitCode = code
			}
			// Store the value in the destination slot before exiting
			if val, ok := resultMap["value"]; ok {
				dest := op.Destination
				if d, ok := resultMap["destination"].(string); ok {
					dest = d
				}
				i.logDebug("storing value %v (type: %T) in destination %s", val, val, dest)
				i.Slots[dest] = val
				i.logDebug("after storing, destination %s contains: %v (type: %T)", dest, i.Slots[dest], i.Slots[dest])
				// Print all values
				switch v := val.(type) {
				case []interface{}:
					i.logDebug("printing array of values: %v (type: %T)", v, v)
					for _, item := range v {
						i.logDebug("printing array item: %v (type: %T)", item, item)
						fmt.Print(item)
					}
				default:
					i.logDebug("printing single value: %v (type: %T)", v, v)
					fmt.Print(val)
				}
				os.Stdout.Sync()
				return &ExitError{code: exitCode}
			} else {
				i.logDebug("no value found in result map")
				return &ExitError{code: exitCode}
			}
		}
	}

	// For debug output, escape newlines to make them visible
	debugResult := fmt.Sprintf("%v", result)
	debugResult = strings.ReplaceAll(debugResult, "\n", "\\n")
	i.logDebug("%s = %s", op.Destination, debugResult)

	// Store the result in the destination slot
	i.logDebug("storing result %v in destination %s", result, op.Destination)
	i.Slots[op.Destination] = result
	return nil
}

// ParseFile parses all instructions from a file
func ParseFile(content string, scriptDir string) ([]*Instruction, error) {
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

func parseLogLevel(level string) int {
	switch strings.ToLower(level) {
	case "debug":
		return log.Debug
	case "info":
		return log.Info
	case "warn":
		return log.Warn
	default:
		return log.Error
	}
}

func main() {
	logLevelFlag := flag.String("log-level", "error", "Set log level: debug, info, error")
	interactiveFlag := flag.Bool("interactive", false, "Run in interactive mode")
	inputFlag := flag.String("input", "", "Input for the script")
	flag.Parse()

	log.Level = parseLogLevel(*logLevelFlag)

	args := flag.Args()
	if len(args) < 1 {
		fmt.Println("Usage: gnd [--log-level debug|info|error] [--interactive] [--input <text>] <file-path>")
		os.Exit(1)
	}

	scriptPath := args[0]
	scriptDir := filepath.Dir(scriptPath)

	contentBytes, err := os.ReadFile(scriptPath)
	if err != nil {
		log.Printf(log.Error, "[ERROR] Error reading file: %v", err)
		os.Exit(1)
	}
	content := string(contentBytes)

	instructions, err := ParseFile(content, scriptDir)
	if err != nil {
		log.Printf(log.Error, "[ERROR] Error parsing file: %v", err)
		os.Exit(1)
	}

	interpreter := NewInterpreter(scriptDir)

	// Handle interactive mode
	if *interactiveFlag {
		if *inputFlag != "" {
			log.Printf(log.Error, "[ERROR] Cannot use --input with --interactive mode")
			os.Exit(1)
		}
		reader := bufio.NewReader(os.Stdin)
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Printf(log.Error, "[ERROR] Error reading input: %v", err)
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
				log.Printf(log.Error, "[ERROR] Input is required. Either provide it through stdin or use --input parameter")
				os.Exit(1)
			}
		}
	}

	var lastResult interface{}

	for i, op := range instructions {
		err = interpreter.ExecuteInstruction(op, i)
		if err != nil {
			interpreter.logDebug("got error from ExecuteInstruction: %v (%T)", err, err)
			if exitErr, ok := err.(*ExitError); ok {
				interpreter.logDebug("error is an ExitError")
				// Print the last result before exiting
				if op != nil {
					interpreter.logDebug("handling exit signal, destination=%s", op.Destination)
					if val, ok := interpreter.Slots[op.Destination]; ok {
						interpreter.logDebug("found value %v in destination %s, printing it", val, op.Destination)
						fmt.Print(val)
						os.Stdout.Sync()
					} else {
						interpreter.logDebug("no value found in destination %s", op.Destination)
						// Try to print from _ as a fallback
						if val, ok := interpreter.Slots["_"]; ok {
							interpreter.logDebug("found value %v in _, printing it", val)
							fmt.Print(val)
							os.Stdout.Sync()
						}
					}
				}
				os.Exit(exitErr.code)
			} else {
				interpreter.logDebug("error is not an ExitError")
			}
			log.Printf(log.Error, "[ERROR] Error executing instruction %d: %v", i+1, err)
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
