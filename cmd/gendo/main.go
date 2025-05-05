package main

import (
	"fmt"
	"os"

	"github.com/hyperifyio/gnd/core/primitive"
)

func main() {
	// Initialize with a default world token
	worldToken := "init"

	// For now, just test the file read primitive
	if len(os.Args) > 1 {
		content, newWorldToken, err := primitive.FileRead(worldToken, os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Content: %s\n", content)
		fmt.Printf("New world token: %s\n", newWorldToken)
	} else {
		fmt.Println("Usage: gendo <file-path>")
		os.Exit(1)
	}
} 