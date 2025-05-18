package helpers

import (
	"path/filepath"
	"strings"
)

// SubroutinePath constructs the path to a subroutine file
// - name is the opcode name, which can be a relative or absolute path.
// - dir is the directory of the currently executing main script.
func SubroutinePath(name, dir string) string {
	subPath := name
	if !strings.HasPrefix(name, "/") {
		subPath = filepath.Join(dir, name)
	}
	if !strings.HasSuffix(name, ".gnd") {
		subPath += ".gnd"
	}
	return subPath
}
