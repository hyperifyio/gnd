package embedded_routines

import (
	"embed"
	"io/fs"
	"path/filepath"
	"strings"

	"github.com/hyperifyio/gnd/pkg/primitive_services"
)

//go:embed rootfs
var rootfs embed.FS

// GetEmbeddedRoutinesFS returns the embedded filesystem containing /gnd/*
func GetEmbeddedRoutinesFS() fs.FS {
	fsys, err := fs.Sub(rootfs, "rootfs")
	if err != nil {
		panic(err)
	}
	return fsys
}

// GetAliasesFromFS returns a map of aliases to their corresponding paths from the filesystem
func GetAliasesFromFS(fsys fs.FS) (map[string]string, error) {
	aliases := make(map[string]string)
	err := fs.WalkDir(fsys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".gnd" {
			base := filepath.Base(path)
			alias := strings.TrimSuffix(base, ".gnd")
			aliases[alias] = "/" + path
		}
		return nil
	})
	return aliases, err
}

func init() {
	fsys := GetEmbeddedRoutinesFS()
	aliases, err := GetAliasesFromFS(fsys)
	if err != nil {
		panic(err)
	}

	for alias, path := range aliases {
		primitive_services.RegisterOpcodeAlias(alias, path)
	}
}
