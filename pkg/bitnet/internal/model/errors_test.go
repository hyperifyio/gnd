package model

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestErrorDefinitions verifies that all error definitions are properly set up
// and can be used for error checking.
func TestErrorDefinitions(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		message string
	}{
		// Filesystem errors
		{
			name:    "ErrFSNotSet",
			err:     ErrFSNotSet,
			message: "filesystem cannot be nil",
		},
		{
			name:    "ErrPathEmpty",
			err:     ErrPathEmpty,
			message: "model path cannot be empty",
		},
		// Model loader errors
		{
			name:    "ErrModelNotFound",
			err:     ErrModelNotFound,
			message: "model file not found",
		},
		{
			name:    "ErrInvalidGGUF",
			err:     ErrInvalidGGUF,
			message: "invalid GGUF magic number",
		},
		{
			name:    "ErrModelNotSet",
			err:     ErrModelNotSet,
			message: "model path not set",
		},
		{
			name:    "ErrReaderNil",
			err:     ErrReaderNil,
			message: "reader is nil",
		},
		// Tokenizer errors
		{
			name:    "ErrTokenizerNotFound",
			err:     ErrTokenizerNotFound,
			message: "tokenizer file not found",
		},
		{
			name:    "ErrVocabNotLoaded",
			err:     ErrVocabNotLoaded,
			message: "vocabulary not loaded",
		},
		{
			name:    "ErrUnknownToken",
			err:     ErrUnknownToken,
			message: "unknown token encountered",
		},
		{
			name:    "ErrUnknownTokenID",
			err:     ErrUnknownTokenID,
			message: "unknown token ID",
		},
		{
			name:    "ErrDecodeFailed",
			err:     ErrDecodeFailed,
			message: "failed to decode tokenizer file",
		},
		{
			name:    "ErrSequenceTooLong",
			err:     ErrSequenceTooLong,
			message: "token sequence exceeds maximum length",
		},
		{
			name:    "ErrVocabRead",
			err:     ErrVocabRead,
			message: "failed to read vocabulary file",
		},
		{
			name:    "ErrVocabParse",
			err:     ErrVocabParse,
			message: "failed to parse vocabulary file",
		},
		{
			name:    "ErrMergesRead",
			err:     ErrMergesRead,
			message: "failed to read merges file",
		},
		{
			name:    "ErrSpecialRead",
			err:     ErrSpecialRead,
			message: "failed to read special tokens file",
		},
		{
			name:    "ErrSpecialParse",
			err:     ErrSpecialParse,
			message: "failed to parse special tokens file",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test error message
			assert.Equal(t, tt.message, tt.err.Error())

			// Test error type
			assert.True(t, errors.Is(tt.err, tt.err))

			// Test error wrapping
			wrappedErr := errors.New("wrapped: " + tt.err.Error())
			assert.False(t, errors.Is(wrappedErr, tt.err))
		})
	}
}

// TestErrorUniqueness verifies that all error definitions are unique
// and not aliases of each other.
func TestErrorUniqueness(t *testing.T) {
	allErrors := []error{
		// Filesystem errors
		ErrFSNotSet,
		ErrPathEmpty,
		// Model loader errors
		ErrModelNotFound,
		ErrInvalidGGUF,
		ErrModelNotSet,
		ErrReaderNil,
		// Tokenizer errors
		ErrTokenizerNotFound,
		ErrVocabNotLoaded,
		ErrUnknownToken,
		ErrUnknownTokenID,
		ErrDecodeFailed,
		ErrSequenceTooLong,
		ErrVocabRead,
		ErrVocabParse,
		ErrMergesRead,
		ErrSpecialRead,
		ErrSpecialParse,
	}

	// Check that each error is unique
	for i, err1 := range allErrors {
		for j, err2 := range allErrors {
			if i != j {
				assert.False(t, errors.Is(err1, err2),
					"Error %v should not be an alias of %v", err1, err2)
			}
		}
	}
}

// TestErrorUsage demonstrates how to use these errors in practice
// and verifies that error checking works as expected.
func TestErrorUsage(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		checkErr error
		wantIs   bool
	}{
		{
			name:     "exact match",
			err:      ErrModelNotFound,
			checkErr: ErrModelNotFound,
			wantIs:   true,
		},
		{
			name:     "different errors",
			err:      ErrModelNotFound,
			checkErr: ErrTokenizerNotFound,
			wantIs:   false,
		},
		{
			name:     "wrapped error",
			err:      errors.New("wrapped: " + ErrModelNotFound.Error()),
			checkErr: ErrModelNotFound,
			wantIs:   false,
		},
		{
			name:     "filesystem error",
			err:      ErrFSNotSet,
			checkErr: ErrFSNotSet,
			wantIs:   true,
		},
		{
			name:     "tokenizer error",
			err:      ErrUnknownToken,
			checkErr: ErrUnknownToken,
			wantIs:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.wantIs, errors.Is(tt.err, tt.checkErr))
		})
	}
}

// TestErrorMessages verifies that error messages are properly formatted
// and contain the expected information.
func TestErrorMessages(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		message string
	}{
		{
			name:    "filesystem error",
			err:     ErrFSNotSet,
			message: "filesystem cannot be nil",
		},
		{
			name:    "model loader error",
			err:     ErrModelNotFound,
			message: "model file not found",
		},
		{
			name:    "tokenizer error",
			err:     ErrUnknownToken,
			message: "unknown token encountered",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errMsg := tt.err.Error()
			assert.Equal(t, tt.message, errMsg)
		})
	}
}

// TestErrorCategories verifies that errors are properly categorized
// and grouped by their functional area.
func TestErrorCategories(t *testing.T) {
	tests := []struct {
		name     string
		category string
		errors   []error
	}{
		{
			name:     "filesystem errors",
			category: "filesystem",
			errors:   []error{ErrFSNotSet, ErrPathEmpty},
		},
		{
			name:     "model loader errors",
			category: "model loader",
			errors:   []error{ErrModelNotFound, ErrInvalidGGUF, ErrModelNotSet, ErrReaderNil},
		},
		{
			name:     "tokenizer errors",
			category: "tokenizer",
			errors: []error{
				ErrTokenizerNotFound, ErrVocabNotLoaded, ErrUnknownToken,
				ErrUnknownTokenID, ErrDecodeFailed, ErrSequenceTooLong,
				ErrVocabRead, ErrVocabParse, ErrMergesRead,
				ErrSpecialRead, ErrSpecialParse,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify that all errors in the category are unique
			for i, err1 := range tt.errors {
				for j, err2 := range tt.errors {
					if i != j {
						assert.False(t, errors.Is(err1, err2),
							"Error %v should not be an alias of %v in category %s",
							err1, err2, tt.category)
					}
				}
			}

			// Verify that errors from different categories are not aliases
			for _, err1 := range tt.errors {
				for _, category := range tests {
					if category.name != tt.name {
						for _, err2 := range category.errors {
							assert.False(t, errors.Is(err1, err2),
								"Error %v from category %s should not be an alias of %v from category %s",
								err1, tt.category, err2, category.category)
						}
					}
				}
			}
		})
	}
}
