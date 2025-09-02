package llm

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"
)

// Helper function to check if Ollama is available (used by compatibility tests)
func isOllamaCompatibilityAvailable() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://localhost:11434/api/tags")
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

// TestProviderCompatibility tests that all available providers follow the same API
func TestProviderCompatibility(t *testing.T) {
	providers := []struct {
		name      string
		createFn  func() (Client, error)
		available bool
	}{
		{
			name: "OpenAI",
			createFn: func() (Client, error) {
				apiKey := os.Getenv("OPENAI_API_KEY")
				if apiKey == "" {
					return nil, fmt.Errorf("API key not available")
				}
				factory := NewFactory()
				return factory.CreateClient(ClientConfig{
					Provider: "openai",
					APIKey:   apiKey,
					Model:    "gpt-4o-mini",
				})
			},
			available: os.Getenv("OPENAI_API_KEY") != "",
		},
		{
			name: "Gemini",
			createFn: func() (Client, error) {
				apiKey := os.Getenv("GEMINI_API_KEY")
				if apiKey == "" {
					return nil, fmt.Errorf("API key not available")
				}
				factory := NewFactory()
				return factory.CreateClient(ClientConfig{
					Provider: "gemini",
					APIKey:   apiKey,
					Model:    "gemini-1.5-flash",
				})
			},
			available: os.Getenv("GEMINI_API_KEY") != "",
		},
		{
			name: "Ollama",
			createFn: func() (Client, error) {
				if !isOllamaCompatibilityAvailable() {
					return nil, fmt.Errorf("server not available")
				}
				factory := NewFactory()
				return factory.CreateClient(ClientConfig{
					Provider: "ollama",
					Model:    "llama3.2:3b",
				})
			},
			available: isOllamaCompatibilityAvailable(),
		},
		{
			name: "OpenRouter",
			createFn: func() (Client, error) {
				apiKey := os.Getenv("OPENROUTER_API_KEY")
				if apiKey == "" {
					return nil, fmt.Errorf("API key not available")
				}
				factory := NewFactory()
				return factory.CreateClient(ClientConfig{
					Provider: "openrouter",
					APIKey:   apiKey,
					Model:    "openai/gpt-3.5-turbo",
				})
			},
			available: os.Getenv("OPENROUTER_API_KEY") != "",
		},
	}

	availableProviders := 0
	for _, provider := range providers {
		if provider.available {
			availableProviders++
		}
	}

	if availableProviders == 0 {
		t.Skip("No LLM providers available for compatibility testing")
	}

	t.Logf("Testing compatibility across %d available providers", availableProviders)

	for _, provider := range providers {
		if !provider.available {
			t.Logf("Skipping %s - not available", provider.name)
			continue
		}

		t.Run(fmt.Sprintf("Compatibility_%s", provider.name), func(t *testing.T) {
			client, err := provider.createFn()
			if err != nil {
				t.Skipf("Failed to create %s client: %v", provider.name, err)
			}
			defer func() { _ = client.Close() }()

			// Test that all clients implement the same interface
			info := client.GetModelInfo()
			if info.Provider == "" {
				t.Errorf("%s: Provider should not be empty", provider.name)
			}
			if info.Name == "" {
				t.Errorf("%s: Model name should not be empty", provider.name)
			}
			if info.MaxTokens <= 0 {
				t.Errorf("%s: MaxTokens should be positive, got %d", provider.name, info.MaxTokens)
			}

			t.Logf("✓ %s: %s (%d max tokens, tools: %t)",
				provider.name, info.Name, info.MaxTokens, info.SupportsTools)

			// Test Close method
			if err := client.Close(); err != nil {
				t.Errorf("%s: Close() returned error: %v", provider.name, err)
			}
		})
	}
}

// TestProviderFactoryCompatibility tests provider creation through the factory
func TestProviderFactoryCompatibility(t *testing.T) {
	factory := NewFactory()

	t.Run("OpenAI_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "openai",
			Model:    "gpt-4o-mini",
			APIKey:   "test-key",
		}

		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create OpenAI client through factory: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})

	t.Run("Gemini_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "gemini",
			Model:    "gemini-1.5-flash",
			APIKey:   "test-key",
		}

		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create Gemini client through factory: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})

	t.Run("Ollama_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "ollama",
			Model:    "llama3.2:3b",
		}

		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create Ollama client through factory: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})

	t.Run("OpenRouter_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "openrouter",
			Model:    "openai/gpt-3.5-turbo",
			APIKey:   "test-key",
		}

		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create OpenRouter client through factory: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})

	t.Run("CustomBaseURL_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "openai",
			Model:    "gpt-4o-mini",
			APIKey:   "test-key",
			BaseURL:  "https://custom-endpoint.example.com",
		}

		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create client with custom base URL through factory: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})

	t.Run("UnsupportedProvider_Factory", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "nonexistent-provider",
			Model:    "some-model",
			APIKey:   "test-key",
		}

		_, err := factory.CreateClient(config)
		if err == nil {
			t.Error("Expected error for unsupported provider")
		}

		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code != "unsupported_provider" {
				t.Errorf("Expected code 'unsupported_provider', got '%s'", llmErr.Code)
			}
		}
	})
}

// Removed TestConsolidatedLLMInterface since it tested the deleted bridge functions to core.ModelConfig

// Removed TestStandardizedErrorHandling and TestConvenienceFunctions since they used deleted bridge/convenience functions

func TestMockClientIntegration(t *testing.T) {
	t.Parallel()
	t.Run("MockClientWithChatCompletion", func(t *testing.T) {
		t.Parallel()
		mockClient := NewMockClient("test-model", "mock")
		mockClient.WithSimpleResponse("Hello from mock!")

		req := ChatRequest{
			Model: "test-model",
			Messages: []Message{
				{Role: RoleUser, Content: []MessageContent{NewTextContent("Test message")}},
			},
		}

		resp, err := mockClient.ChatCompletion(context.Background(), req)
		if err != nil {
			t.Errorf("Mock chat completion failed: %v", err)
		}

		if resp == nil || len(resp.Choices) == 0 {
			t.Error("Expected response with choices")
		} else {
			if resp.Choices[0].Message.GetText() != "Hello from mock!" {
				t.Errorf("Expected 'Hello from mock!', got '%s'", resp.Choices[0].Message.Content)
			}
		}

		// Verify call logging
		calls := mockClient.GetCallLog()
		if len(calls) != 1 {
			t.Errorf("Expected 1 call logged, got %d", len(calls))
		}

		_ = mockClient.Close()
	})
}
