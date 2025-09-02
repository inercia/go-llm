package llm

import (
	"fmt"
	"strings"
)

// Factory creates LLM clients based on configuration
type Factory struct{}

// NewFactory creates a new client factory
func NewFactory() *Factory {
	return &Factory{}
}

// CreateClient creates an LLM client based on the configuration
func (f *Factory) CreateClient(config ClientConfig) (Client, error) {
	// Default to "openai" if provider is empty for backward compatibility
	provider := config.Provider
	if provider == "" {
		provider = "openai"
	}
	provider = strings.ToLower(provider)

	// Validate required fields
	if config.Model == "" {
		return nil, &Error{
			Code:    "missing_model",
			Message: "model is required",
			Type:    "validation_error",
		}
	}
	switch provider {
	case "openai":
		return NewOpenAIClient(config)
	case "gemini":
		return NewGeminiClient(config)
	case "ollama":
		return NewOllamaClient(config)
	case "openrouter":
		return NewOpenRouterClient(config)
	case "anthropic":
		return nil, &Error{
			Code:    "not_implemented",
			Message: "Anthropic client not yet implemented",
			Type:    "implementation_error",
		}
	case "mocked", "mock":
		// For testing purposes, create a mock client
		return NewMockClient(config.Model, "mock"), nil
	default:
		return nil, &Error{
			Code:    "unsupported_provider",
			Message: fmt.Sprintf("unsupported provider: %s", provider),
			Type:    "validation_error",
		}
	}
}
