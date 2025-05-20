package config

import (
	"runtime"
	"testing"
)

func TestNewRuntimeConfig(t *testing.T) {
	cfg := NewRuntimeConfig()
	if cfg.MaxProcs != runtime.NumCPU() {
		t.Errorf("MaxProcs = %d, want %d", cfg.MaxProcs, runtime.NumCPU())
	}
}

func TestValidate(t *testing.T) {
	cfg := &RuntimeConfig{MaxProcs: 4}
	if err := cfg.Validate(); err != nil {
		t.Errorf("Validate() returned error: %v", err)
	}
}
