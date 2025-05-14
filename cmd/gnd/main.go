package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/hyperifyio/gnd/pkg/core"
	"github.com/hyperifyio/gnd/pkg/log"
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
		log.Level = log.Debug
	}

	if len(flag.Args()) < 1 {
		fmt.Fprintln(os.Stderr, "Error: missing script file")
		printHelp()
		os.Exit(1)
	}

	scriptPath := flag.Args()[0]
	scriptDir := filepath.Dir(scriptPath)

	// Parse the script file
	instructions, err := core.ParseFile(scriptPath)
	if err != nil {
		fmt.Printf("Error parsing script: %v\n", err)
		os.Exit(1)
	}

	// Create a new core
	interpreterImpl := core.NewInterpreter(scriptDir)

	// Execute each instruction
	status := 0
	var value interface{}
	if value, err = interpreterImpl.ExecuteInstructions(instructions); err != nil {
		if exitErr, ok := core.GetExitResult(err); ok {
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
