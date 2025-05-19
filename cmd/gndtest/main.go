package main

import (
	"fmt"
	"os"
	"strings"
)

// TestCase represents a single test case
type TestCase struct {
	Name        string
	Input       string
	Expected    string
	Description string
}

// TestResult represents the result of running a test
type TestResult struct {
	TestCase TestCase
	Passed   bool
	Output   string
	Error    string
}

func parseTestFile(content string) (TestCase, error) {
	lines := strings.Split(content, "\n")
	if len(lines) < 4 {
		return TestCase{}, fmt.Errorf("invalid test file format")
	}

	// Format:
	// name: Test Name
	// description: Test Description
	// input: |
	//   test input
	// expected: |
	//   expected output
	var testCase TestCase
	var currentSection string
	var currentContent strings.Builder

	for _, line := range lines {
		if strings.HasPrefix(line, "name: ") {
			testCase.Name = strings.TrimPrefix(line, "name: ")
		} else if strings.HasPrefix(line, "description: ") {
			testCase.Description = strings.TrimPrefix(line, "description: ")
		} else if strings.HasPrefix(line, "input: |") {
			currentSection = "input"
			currentContent.Reset()
		} else if strings.HasPrefix(line, "expected: |") {
			currentSection = "expected"
			currentContent.Reset()
		} else if currentSection != "" {
			if line == "" && currentContent.Len() == 0 {
				continue
			}
			currentContent.WriteString(line)
			currentContent.WriteString("\n")
		}
	}

	testCase.Input = strings.TrimSpace(currentContent.String())
	return testCase, nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: gndtest <test-file>")
		os.Exit(1)
	}

	testFile := os.Args[1]

	// Read test file
	content, err := os.ReadFile(testFile)
	if err != nil {
		fmt.Printf("Error reading test file: %v\n", err)
		os.Exit(1)
	}

	// Parse test case
	testCase, err := parseTestFile(string(content))
	if err != nil {
		fmt.Printf("Error parsing test case: %v\n", err)
		os.Exit(1)
	}

	// Run test
	result := runTest(testCase)

	// Print result
	if result.Passed {
		fmt.Printf("✓ Test '%s' passed\n", testCase.Name)
	} else {
		fmt.Printf("✗ Test '%s' failed\n", testCase.Name)
		if result.Error != "" {
			fmt.Printf("Error: %s\n", result.Error)
		}
		if result.Output != "" {
			fmt.Printf("Output: %s\n", result.Output)
		}
		os.Exit(1)
	}
}

func runTest(testCase TestCase) TestResult {
	// TODO: Implement actual test execution using gnd
	// For now, just return a placeholder result
	return TestResult{
		TestCase: testCase,
		Passed:   true,
	}
}
