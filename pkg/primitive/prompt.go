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

// Prompt represents the prompt primitive
type Prompt struct{}

func (p *Prompt) Name() string {
	return "/gnd/prompt"
}

func (p *Prompt) Execute(args []interface{}) (interface{}, error) {
	if len(args) < 1 {
		return nil, fmt.Errorf("prompt expects at least 1 argument, got %d", len(args))
	}

	prompt := fmt.Sprintf("%v", args[0])
	for i := 1; i < len(args); i++ {
		prompt += " " + fmt.Sprintf("%v", args[i])
	}

	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	// Skip actual API call in test environment
	if os.Getenv("GO_TEST") == "1" {
		return fmt.Sprintf("Echo: %s", prompt), nil
	}

	client := &http.Client{
		Timeout: DefaultConfig().Timeout,
	}

	return LLMImpl(client, DefaultConfig(), apiKey, prompt)
}

// LLMImpl sends a prompt to the LLM server and receives a response
func LLMImpl(client LLMClient, config LLMConfig, apiKey string, prompt string) (string, error) {
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

	// Log request details
	fmt.Fprintf(os.Stderr, "[DEBUG] LLM Request:\n")
	fmt.Fprintf(os.Stderr, "[DEBUG] URL: %s\n", config.BaseURL+"/chat/completions")
	fmt.Fprintf(os.Stderr, "[DEBUG] Model: %s\n", config.Model)
	fmt.Fprintf(os.Stderr, "[DEBUG] Temperature: %f\n", config.Temperature)
	fmt.Fprintf(os.Stderr, "[DEBUG] MaxTokens: %d\n", config.MaxTokens)
	fmt.Fprintf(os.Stderr, "[DEBUG] Messages: %+v\n", messages)

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

	// Log response details
	fmt.Fprintf(os.Stderr, "[DEBUG] LLM Response:\n")
	fmt.Fprintf(os.Stderr, "[DEBUG] Status: %s\n", resp.Status)
	fmt.Fprintf(os.Stderr, "[DEBUG] Body: %s\n", string(body))

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
	fmt.Fprintf(os.Stderr, "[DEBUG] Response Content: %s\n", chatResp.Choices[0].Message.Content)

	return chatResp.Choices[0].Message.Content, nil
}

// LLM is a primitive that handles LLM operations
type LLM struct{}

func (l *LLM) Name() string {
	return "/gnd/llm"
}

func (l *LLM) Execute(args []string) (interface{}, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("llm requires at least one argument")
	}

	// Check for API key first
	if os.Getenv("OPENAI_API_KEY") == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is required")
	}

	// In test mode, just echo the input
	if os.Getenv("GO_TEST") == "1" {
		return fmt.Sprintf("Echo: %s", args[0]), nil
	}

	// TODO: Implement actual LLM call
	return nil, fmt.Errorf("LLM implementation not yet complete")
}
