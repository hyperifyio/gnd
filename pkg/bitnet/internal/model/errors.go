package model

import "errors"

var (
	// Filesystem errors
	ErrFSNotSet  = errors.New("filesystem cannot be nil")
	ErrPathEmpty = errors.New("model path cannot be empty")

	// Model loader errors
	ErrModelNotFound = errors.New("model file not found")
	ErrInvalidGGUF   = errors.New("invalid GGUF magic number")
	ErrModelNotSet   = errors.New("model path not set")
	ErrReaderNil     = errors.New("reader is nil")

	// Tokenizer errors
	ErrTokenizerNotFound = errors.New("tokenizer file not found")
	ErrVocabNotLoaded    = errors.New("vocabulary not loaded")
	ErrUnknownToken      = errors.New("unknown token encountered")
	ErrUnknownTokenID    = errors.New("unknown token ID")
	ErrDecodeFailed      = errors.New("failed to decode tokenizer file")
	ErrSequenceTooLong   = errors.New("token sequence exceeds maximum length")
	ErrVocabRead         = errors.New("failed to read vocabulary file")
	ErrVocabParse        = errors.New("failed to parse vocabulary file")
	ErrMergesRead        = errors.New("failed to read merges file")
	ErrSpecialRead       = errors.New("failed to read special tokens file")
	ErrSpecialParse      = errors.New("failed to parse special tokens file")
)
