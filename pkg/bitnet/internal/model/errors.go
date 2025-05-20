package model

import "errors"

var (
	// Common errors
	ErrFSNotSet      = errors.New("filesystem cannot be nil")
	ErrPathEmpty     = errors.New("model path cannot be empty")
	ErrModelNotFound = errors.New("model file not found")
	ErrModelNotSet   = errors.New("model path not set")
	ErrReaderNil     = errors.New("reader is nil")
	ErrChunkRead     = errors.New("failed to read model chunk")

	// GGUF specific errors
	ErrInvalidGGUF = errors.New("invalid GGUF magic number")

	// Tokenizer specific errors
	ErrTokenizerNotFound = errors.New("tokenizer file not found")
	ErrVocabNotLoaded    = errors.New("vocabulary not loaded")
	ErrUnknownToken      = errors.New("unknown token")
	ErrUnknownTokenID    = errors.New("unknown token ID")
	ErrDecodeFailed      = errors.New("failed to decode tokenizer file")
	ErrBPEFailed         = errors.New("failed to apply BPE algorithm")
)
