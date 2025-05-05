package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive"
)

// Opcode represents a Gendo instruction
type Opcode struct {
	Name      string
	Args      []string
	ReturnVal string
}

// World represents the execution environment
type World struct {
	Slots      map[string]interface{}
	WorldToken string
}

// RegisterGoPrimitives registers all Go primitives under /go/ namespace
func RegisterGoPrimitives() map[string]func(*World, []string) (interface{}, error) {
	primitives := make(map[string]func(*World, []string) (interface{}, error))

	// File operations
	primitives["/go/file_read"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("file_read expects 2 arguments")
		}
		content, newWorldToken, err := primitive.FileRead(args[0], args[1])
		if err != nil {
			return nil, err
		}
		w.WorldToken = newWorldToken
		return content, nil
	}

	primitives["/go/file_list"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("file_list expects 2 arguments")
		}
		names, newWorldToken, err := primitive.FileList(args[0], args[1])
		if err != nil {
			return nil, err
		}
		w.WorldToken = newWorldToken
		return names, nil
	}

	primitives["/go/emit_file"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 3 {
			return nil, fmt.Errorf("emit_file expects 3 arguments")
		}
		newWorldToken, err := primitive.EmitFile(args[0], args[1], args[2])
		if err != nil {
			return nil, err
		}
		w.WorldToken = newWorldToken
		return newWorldToken, nil
	}

	// String operations
	primitives["/go/string_split"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("string_split expects 2 arguments")
		}
		parts := primitive.StringSplit(args[0], args[1])
		return parts, nil
	}

	primitives["/go/string_match"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("string_match expects 2 arguments")
		}
		return primitive.StringMatch(args[0], args[1])
	}

	// List operations
	primitives["/go/list_map"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("list_map expects 2 arguments")
		}
		return primitive.ListMap(args[0], args[1])
	}

	primitives["/go/list_filter"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("list_filter expects 2 arguments")
		}
		return primitive.ListFilter(args[0], args[1])
	}

	primitives["/go/list_fold"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 3 {
			return nil, fmt.Errorf("list_fold expects 3 arguments")
		}
		return primitive.ListFold(args[0], args[1], args[2])
	}

	// Dictionary operations
	primitives["/go/dict_get"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("dict_get expects 2 arguments")
		}
		return primitive.DictGet(args[0], args[1])
	}

	primitives["/go/dict_set"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 3 {
			return nil, fmt.Errorf("dict_set expects 3 arguments")
		}
		return primitive.DictSet(args[0], args[1], args[2])
	}

	// Other operations
	primitives["/go/concat"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("concat expects 2 arguments")
		}
		return primitive.Concat(args[0], args[1])
	}

	primitives["/go/format"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("format expects 2 arguments")
		}
		return primitive.Format(args[0], args[1])
	}

	primitives["/go/parse_number"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("parse_number expects 1 argument")
		}
		return primitive.ParseNumber(args[0])
	}

	primitives["/go/llm_call"] = func(w *World, args []string) (interface{}, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("llm_call expects 2 arguments")
		}
		return primitive.LLMCall(args[0], args[1])
	}

	return primitives
}

// ParseInstruction parses a single Gendo instruction
func ParseInstruction(line string) (*Opcode, error) {
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid instruction: %s", line)
	}

	return &Opcode{
		Name:      parts[0],
		ReturnVal: parts[1],
		Args:      parts[2:],
	}, nil
}

// ExecuteInstruction executes a single Gendo instruction
func ExecuteInstruction(w *World, op *Opcode, primitives map[string]func(*World, []string) (interface{}, error)) error {
	prim, ok := primitives[op.Name]
	if !ok {
		return fmt.Errorf("unknown opcode: %s", op.Name)
	}

	result, err := prim(w, op.Args)
	if err != nil {
		return err
	}

	w.Slots[op.ReturnVal] = result
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: gendo <file-path>")
		os.Exit(1)
	}

	// Initialize world
	world := &World{
		Slots:      make(map[string]interface{}),
		WorldToken: "init",
	}

	// Register Go primitives
	primitives := RegisterGoPrimitives()

	// Read and execute the Gendo file
	content, _, err := primitive.FileRead(world.WorldToken, os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	// Split into lines and execute each instruction
	lines := strings.Split(content, "\n")
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		op, err := ParseInstruction(line)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error on line %d: %v\n", i+1, err)
			os.Exit(1)
		}

		err = ExecuteInstruction(world, op, primitives)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error executing line %d: %v\n", i+1, err)
			os.Exit(1)
		}
	}
}
