package factory

import (
	"fmt"
	"strings"

	"github.com/inercia/go-llm/pkg/llm"
)

const DefaultProvider = "openai"

// Factory creates LLM clients based on configuration
type Factory struct{}

// New creates a new client factory
func New() *Factory {
	return &Factory{}
}

// CreateClient creates an LLM client based on the configuration
func (f *Factory) CreateClient(config llm.ClientConfig) (llm.Client, error) {
	// Default to "openai" if provider is empty for backward compatibility
	provider := config.Provider
	if provider == "" {
		provider = DefaultProvider
	}
	provider = strings.ToLower(provider)

	// Validate required fields
	if config.Model == "" {
		return nil, &llm.Error{
			Code:    "missing_model",
			Message: "model is required",
			Type:    "validation_error",
		}
	}
	// Use the provider registry to create clients
	constructor, exists := GetProvider(provider)
	if !exists {
		return nil, &llm.Error{
			Code:    "unsupported_provider",
			Message: fmt.Sprintf("unsupported provider: %s", provider),
			Type:    "validation_error",
		}
	}

	return constructor(config)
}
