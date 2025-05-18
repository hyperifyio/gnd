package embedded_routines_test

import (
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/hyperifyio/gnd/pkg/embedded_routines"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/stretchr/testify/assert"
)

func TestGetUnitsFS(t *testing.T) {
	fsys := embedded_routines.GetEmbeddedRoutinesFS()
	assert.NotNil(t, fsys, "GetEmbeddedRoutinesFS should return a non-nil filesystem")

	// Test that we can read the gnd directory
	entries, err := fs.ReadDir(fsys, "gnd")
	assert.NoError(t, err, "Should be able to read gnd directory")
	assert.NotEmpty(t, entries, "gnd directory should not be empty")

	// Verify that all entries are .gnd files
	for _, entry := range entries {
		assert.False(t, entry.IsDir(), "All entries should be files")
		assert.Equal(t, ".gnd", filepath.Ext(entry.Name()), "All files should have .gnd extension")
	}
}

func TestGetAliasesFromFS(t *testing.T) {
	fsys := embedded_routines.GetEmbeddedRoutinesFS()
	aliases, err := embedded_routines.GetAliasesFromFS(fsys)
	assert.NoError(t, err, "Should be able to get aliases from filesystem")
	assert.NotEmpty(t, aliases, "Should have found aliases")

	// Verify specific known aliases
	expectedAliases := map[string]string{
		"debug":   "/gnd/debug.gnd",
		"error":   "/gnd/error.gnd",
		"info":    "/gnd/info.gnd",
		"let":     "/gnd/let.gnd",
		"println": "/gnd/println.gnd",
		"warn":    "/gnd/warn.gnd",
	}

	assert.Equal(t, expectedAliases, aliases, "Aliases should match expected values")

	// Verify that each alias points to a valid file
	for alias, path := range aliases {
		// Remove leading slash for fs.Stat
		filePath := strings.TrimPrefix(path, "/")
		_, err := fs.Stat(fsys, filePath)
		assert.NoError(t, err, "Alias %s should point to a valid file", alias)
	}
}

func TestOpcodeAliasRegistration(t *testing.T) {
	// Get all registered aliases
	aliases := primitive_services.GetDefaultOpcodeMap()
	assert.NotEmpty(t, aliases, "Should have registered aliases")

	// Get expected aliases from filesystem
	fsys := embedded_routines.GetEmbeddedRoutinesFS()
	expectedAliases, err := embedded_routines.GetAliasesFromFS(fsys)
	assert.NoError(t, err, "Should be able to get expected aliases")

	// Verify that all expected aliases are registered
	for alias, path := range expectedAliases {
		registeredPath, exists := aliases[alias]
		assert.True(t, exists, "Expected alias %s should be registered", alias)
		assert.Equal(t, path, registeredPath, "Alias %s should point to correct path", alias)
	}
}
