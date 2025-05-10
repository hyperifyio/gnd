package core

import (
	"github.com/hyperifyio/gnd/pkg/units"
	"io/fs"
	"os"
	"path/filepath"
)

// ResolveSubroutinePath checks if the opcode is a subroutine and returns its path
func ResolveSubroutinePath(opcode string, scriptDir string) (bool, string) {
	if _, err := os.Stat(filepath.Join(scriptDir, opcode+".gnd")); err == nil {
		return true, opcode + ".gnd"
	}
	if _, err := fs.Stat(units.GetUnitsFS(), opcode+".gnd"); err == nil {
		return true, opcode + ".gnd"
	}
	return false, ""
}
