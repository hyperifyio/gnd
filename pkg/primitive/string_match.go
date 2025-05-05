package primitive

import (
	"fmt"
	"regexp"
	"time"
)

// Match represents a single regex match
type Match struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
}

// StringMatch applies a regular expression to a string
func StringMatch(s string, pattern string) ([]Match, error) {
	// Log the operation for debugging
	fmt.Printf("StringMatch: matching %q against pattern %q\n", s, pattern)

	// Compile the pattern with a timeout to prevent DoS
	done := make(chan struct{})
	var re *regexp.Regexp
	var err error

	go func() {
		re, err = regexp.Compile(pattern)
		close(done)
	}()

	select {
	case <-done:
		// Pattern compiled successfully
	case <-time.After(100 * time.Millisecond):
		return nil, fmt.Errorf("pattern compilation timed out")
	}

	if err != nil {
		fmt.Printf("StringMatch: invalid pattern %q: %v\n", pattern, err)
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}

	// Find all matches
	matches := re.FindAllStringIndex(s, -1)
	if matches == nil {
		fmt.Printf("StringMatch: no matches found\n")
		return []Match{}, nil
	}

	// Convert to Match objects
	result := make([]Match, len(matches))
	for i, m := range matches {
		result[i] = Match{
			Start: m[0],
			End:   m[1],
			Text:  s[m[0]:m[1]],
		}
	}

	fmt.Printf("StringMatch: found %d matches\n", len(result))
	return result, nil
} 