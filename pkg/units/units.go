package units

import (
	"embed"
	"io/fs"
)

//go:embed embedded
var unitsFS embed.FS

// GetUnitsFS returns the embedded filesystem containing /gnd/*
func GetUnitsFS() fs.FS {
	fsys, err := fs.Sub(unitsFS, "embedded")
	if err != nil {
		panic(err)
	}
	return fsys
}
