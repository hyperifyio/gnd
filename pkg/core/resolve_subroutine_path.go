package core

import (
	"io/fs"
	"os"
	"path/filepath"

	"github.com/hyperifyio/gnd/pkg/units"
)

// ResolveSubroutinePath checks if the opcode is a subroutine and returns its path
// It first checks in the script directory and then in the embedded filesystem.
// The scriptDir is the directory of the currently executing main script.
// The opcode can also have a path in it.
func ResolveSubroutinePath(opcode string, scriptDir string) (bool, string) {

	// First check in the script directory
	if _, err := os.Stat(filepath.Join(scriptDir, opcode+".gnd")); err == nil {
		return true, opcode + ".gnd"
	}

	// Then check in the embedded filesystem (which is already rooted at gnd/)
	if _, err := fs.Stat(units.GetUnitsFS(), opcode+".gnd"); err == nil {
		return true, opcode + ".gnd"
	}

	return false, ""
}
