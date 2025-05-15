package core

import (
	"path/filepath"
	"strings"
)

// SubroutinePath constructs the path to a subroutine file
func SubroutinePath(name, dir string) string {
	// Construct the path to the subroutine file
	subPath := name
	if !strings.HasPrefix(name, "/") {
		subPath = filepath.Join(dir, name)
	}
	if !strings.HasSuffix(name, ".gnd") {
		subPath += ".gnd"
	}
	return subPath
}
