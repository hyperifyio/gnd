package prompts

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/hyperifyio/gnd/pkg/loggers"
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

// PromptClient defines the interface for LLM operations
type PromptClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// PromptConfig holds the configuration for LLM calls
type PromptConfig struct {
	BaseURL     string
	Model       string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
}

// DefaultConfig returns a default LLM configuration
func DefaultConfig() PromptConfig {
	return PromptConfig{
		BaseURL:     "http://localhost:18080/v1",
		Model:       "bitnet",
		Temperature: 0.7,
		MaxTokens:   1000,
		Timeout:     120 * time.Second,
	}
}

// PromptClientImpl sends a prompt to the LLM server and receives a response
func PromptClientImpl(client PromptClient, config PromptConfig, apiKey string, prompt string) (string, error) {

	// Validate inputs
	if apiKey == "" {
		return "", fmt.Errorf("empty API key")
	}

	if prompt == "" {
		return "", fmt.Errorf("empty prompt")
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

	// Log the request details
	loggers.Printf(loggers.Debug, "LLM Request:")
	loggers.Printf(loggers.Debug, "URL: %s", config.BaseURL+"/chat/completions")
	loggers.Printf(loggers.Debug, "Model: %s", config.Model)
	loggers.Printf(loggers.Debug, "Temperature: %f", config.Temperature)
	loggers.Printf(loggers.Debug, "MaxTokens: %d", config.MaxTokens)
	loggers.Printf(loggers.Debug, "Messages: %+v", messages)

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

	// Log the response details
	loggers.Printf(loggers.Debug, "LLM Response:")
	loggers.Printf(loggers.Debug, "Status: %s", resp.Status)
	loggers.Printf(loggers.Debug, "Body: %s", string(body))

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

	// Log the final response content
	loggers.Printf(loggers.Debug, "Response Content: %s", chatResp.Choices[0].Message.Content)

	return chatResp.Choices[0].Message.Content, nil
}
