package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hyperifyio/gnd/pkg/interpreters"
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/primitives"
)

func printHelp() {
	fmt.Print(`Usage: gnd [options] <script.gnd>
Options:
  -h, --help      Show this help message and exit
  -v, --verbose   Enable verbose (debug) logging

Arguments:
  <script.gnd>    Path to the GND script to execute

Examples:
  gnd examples/debug.gnd
  gnd --verbose examples/debug.gnd
`)
}

func main() {
	help := flag.Bool("help", false, "Show help")
	h := flag.Bool("h", false, "Show help (shorthand)")
	verbose := flag.Bool("verbose", false, "Enable verbose (debug) logging")
	v := flag.Bool("v", false, "Enable verbose (debug) logging (shorthand)")
	flag.Parse()

	if *help || *h {
		printHelp()
		os.Exit(0)
	}

	if *verbose || *v {
		loggers.Level = loggers.Debug
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintln(os.Stderr, "Error: missing script file")
		printHelp()
		os.Exit(1)
	}

	args := flag.Args()
	scriptPath := args[0]
	scriptDir := filepath.Dir(scriptPath)
	loggers.Printf(loggers.Debug, "script path: %v", scriptPath)
	loggers.Printf(loggers.Debug, "script dir: %v", scriptDir)

	scriptArgs := args[1:]
	loggers.Printf(loggers.Debug, "script args: %v", scriptArgs)

	// Create a new core interpreter
	interpreterImpl := interpreters.NewInterpreter(scriptDir, primitive_services.GetDefaultOpcodeMap())
	interpreterImpl.SetSlot("_", scriptArgs)

	// Read the script file
	content, err := os.ReadFile(scriptPath)
	if err != nil {
		fmt.Printf("Error reading the script: %s: %v\n", scriptPath, err)
		os.Exit(1)
	}

	// Parse the instructions
	instructions, err := parsers.ParseInstructionLines(scriptPath, string(content))
	if err != nil {
		fmt.Printf("Error parsing script: %v\n", err)
		os.Exit(1)
	}
	loggers.Printf(loggers.Debug, "loaded %d instructions", len(instructions))

	// Execute each instruction
	status := 0
	var value interface{}
	loggers.Printf(loggers.Debug, "Executing: %s %v", scriptPath, scriptArgs)
	if value, err = interpreterImpl.ExecuteInstructionBlock(scriptPath, scriptArgs, instructions); err != nil {
		if exitErr, ok := primitives.GetExitResult(err); ok {
			value = exitErr.Value
			status = exitErr.Code
		} else {
			fmt.Printf("Error executing instruction: %v\n", err)
			value = nil
			status = 1
		}
	}

	if value != nil {
		fmt.Println(value)
		os.Stdout.Sync()
	}

	os.Exit(status)
}
