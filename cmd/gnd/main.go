package main

import (
	"bufio"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

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
	Slots      map[string]interface{}
	FileSystem primitive.FileSystem
	LLMClient  primitive.LLMClient
	LLMConfig  primitive.LLMConfig
}

// NewInterpreter creates a new interpreter instance
func NewInterpreter(fs primitive.FileSystem, llmClient primitive.LLMClient, llmConfig primitive.LLMConfig) *Interpreter {
	i := &Interpreter{
		Slots:      make(map[string]interface{}),
		FileSystem: fs,
		LLMClient:  llmClient,
		LLMConfig:  llmConfig,
	}
	return i
}

// ParseInstruction parses a single GND instruction
func ParseInstruction(line string) (*Instruction, error) {
	// Skip comments and blank lines
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil
	}

	// Split into tokens
	tokens := strings.Fields(line)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty instruction")
	}

	// Parse opcode
	opcode := tokens[0]
	if !strings.HasPrefix(opcode, "/gnd/") {
		return nil, fmt.Errorf("invalid opcode prefix: %s (must start with /gnd/)", opcode)
	}

	// Parse destination
	var dest string
	var args []string
	if len(tokens) > 1 {
		dest = tokens[1]
		if dest != "_" && !isValidIdentifier(dest) {
			return nil, fmt.Errorf("invalid destination: %s", dest)
		}
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

// isValidIdentifier checks if a string is a valid GND identifier
func isValidIdentifier(s string) bool {
	if len(s) == 0 {
		return false
	}

	// Allow /gnd/ prefix
	if strings.HasPrefix(s, "/gnd/") {
		s = s[5:] // Remove the prefix
	}

	first := s[0]
	if !((first >= 'a' && first <= 'z') || (first >= 'A' && first <= 'Z')) {
		return false
	}
	for _, c := range s[1:] {
		if !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-') {
			return false
		}
	}
	return true
}

// ExecuteInstruction executes a single GND instruction
func (i *Interpreter) ExecuteInstruction(op *Instruction) error {
	if op == nil {
		return nil
	}

	// Resolve arguments from slots if they exist
	resolvedArgs := make([]string, len(op.Arguments))
	for j, arg := range op.Arguments {
		if val, ok := i.Slots[arg]; ok {
			// Convert the slot value to string
			resolvedArgs[j] = fmt.Sprintf("%v", val)
		} else {
			resolvedArgs[j] = arg
		}
	}

	// Get the primitive from the registry
	prim, ok := primitive.Get(op.Opcode)
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Opcode)
	}

	// Execute the primitive
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

	// Create file system implementation
	fs := &OSFileSystem{}

	// Create HTTP client for LLM calls
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Create LLM config
	llmConfig := primitive.DefaultConfig()

	// Initialize interpreter
	interpreter := NewInterpreter(fs, client, llmConfig)

	// Read the GND file
	content, err := primitive.FileRead(fs, os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	// Parse all instructions
	instructions, err := ParseFile(content)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing file: %v\n", err)
		os.Exit(1)
	}

	// Execute all instructions
	for i, op := range instructions {
		err = interpreter.ExecuteInstruction(op)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error executing instruction %d: %v\n", i+1, err)
			os.Exit(1)
		}
	}
}

// OSFileSystem implements primitive.FileSystem using the OS
type OSFileSystem struct{}

func (fs *OSFileSystem) ReadFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (fs *OSFileSystem) ListDir(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	sort.Strings(names)
	return names, nil
}

func (fs *OSFileSystem) WriteFile(path string, content string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}
	return os.WriteFile(path, []byte(content), 0644)
}
