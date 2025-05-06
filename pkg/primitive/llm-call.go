package primitive

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents the request to the LLM API
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	MaxTokens   int       `json:"max_tokens"`
}

// ChatResponse represents the response from the LLM API
type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

// LLMClient defines the interface for LLM operations
type LLMClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// LLMConfig holds the configuration for LLM calls
type LLMConfig struct {
	BaseURL     string
	Model       string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
}

// DefaultConfig returns a default LLM configuration
func DefaultConfig() LLMConfig {
	return LLMConfig{
		BaseURL:     "http://localhost:18080/v1",
		Model:       "bitnet",
		Temperature: 0.7,
		MaxTokens:   1000,
		Timeout:     30 * time.Second,
	}
}

// LLMCall sends a prompt to the LLM server and receives a response
func LLMCall(client LLMClient, config LLMConfig, apiKey string, prompt string) (string, error) {
	// Validate inputs
	if apiKey == "" {
		return "", fmt.Errorf("empty API key")
	}
	if prompt == "" {
		return "", fmt.Errorf("empty prompt")
	}

	// Skip actual API call in test environment
	if os.Getenv("GO_TEST") == "1" {
		return "", fmt.Errorf("API call skipped in test environment")
	}

	messages := []Message{
		{Role: "user", Content: prompt},
	}

	reqBody := ChatRequest{
		Model:       config.Model,
		Messages:    messages,
		Temperature: config.Temperature,
		MaxTokens:   config.MaxTokens,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", config.BaseURL+"/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("API error: %s", body)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no response from API")
	}

	return chatResp.Choices[0].Message.Content, nil
}
