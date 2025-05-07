package units

import (
	"embed"
	"io/fs"
)

//go:embed gnd
var unitsFS embed.FS

// GetUnitsFS returns the embedded filesystem containing GND units
func GetUnitsFS() fs.FS {
	fsys, err := fs.Sub(unitsFS, "gnd")
	if err != nil {
		panic(err)
	}
	return fsys
}
