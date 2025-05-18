package primitives

import (
	"errors"
	"fmt"
	"github.com/hyperifyio/gnd/pkg/helpers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_services"
	"github.com/hyperifyio/gnd/pkg/prompts"
	"net/http"
)

var PromptExpectsAtLeastOneArgument = errors.New("prompt expects at least 1 argument")
var PromptApiKeyNotSet = errors.New("OPENAI_API_KEY environment variable not set")

// Prompt represents the prompt primitive
type Prompt struct{}

func (p *Prompt) Name() string {
	return "/gnd/prompt"
}

func (p *Prompt) Execute(args []interface{}) (interface{}, error) {
	if len(args) < 1 {
		return nil, PromptExpectsAtLeastOneArgument
	}

	prompt, err := parsers.ParseString(args)
	if err != nil {
		return nil, fmt.Errorf("prompt: %v", err)
	}

	// Get API key from environment
	apiKey := helpers.GetEnv("OPENAI_API_KEY", "")
	if apiKey == "" {
		return nil, PromptApiKeyNotSet
	}

	config := prompts.DefaultConfig()
	config.BaseURL = helpers.GetEnv("OPENAI_API_URL", config.BaseURL)
	config.Model = helpers.GetEnv("OPENAI_MODEL", config.Model)

	client := &http.Client{
		Timeout: config.Timeout,
	}

	return prompts.PromptClientImpl(client, config, apiKey, prompt)
}

func init() {
	primitive_services.RegisterPrimitive(&Prompt{})
}
