package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Prompt represents a single LLM prompt
type Prompt struct {
	ID          string
	Content     string
	Description string
}

// SplitPrompt splits a large prompt into smaller LLM-based prompts
func SplitPrompt(content string) ([]Prompt, error) {
	// TODO: Implement actual LLM-based prompt splitting
	// For now, just create a single prompt
	return []Prompt{
		{
			ID:          "main",
			Content:     content,
			Description: "Main prompt",
		},
	}, nil
}

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: gndc <input-file> <output-dir>")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputDir := os.Args[2]

	// Read input file
	content, err := os.ReadFile(inputFile)
	if err != nil {
		fmt.Printf("Error reading input file: %v\n", err)
		os.Exit(1)
	}

	// Split prompt
	prompts, err := SplitPrompt(string(content))
	if err != nil {
		fmt.Printf("Error splitting prompt: %v\n", err)
		os.Exit(1)
	}

	// Create output directory if it doesn't exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Write each prompt to a separate file
	for _, prompt := range prompts {
		outputFile := filepath.Join(outputDir, prompt.ID+".gnd")

		// Format: one instruction per line
		// Each line: opcode destination input1 input2 ...
		instructions := []string{
			"# " + prompt.Description,
			"identity _ _", // Start with identity operation
		}

		// TODO: Process prompt content into actual instructions
		// For now, just add a placeholder instruction
		instructions = append(instructions, "llm-call _ _")

		output := strings.Join(instructions, "\n")
		if err := os.WriteFile(outputFile, []byte(output), 0644); err != nil {
			fmt.Printf("Error writing output file: %v\n", err)
			os.Exit(1)
		}
	}
}
