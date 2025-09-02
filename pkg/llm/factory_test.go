package llm

import (
	"testing"
)

// TestFactory tests the factory functionality
func TestFactory(t *testing.T) {
	t.Parallel()
	t.Run("CreateClient_Validation", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()

		// Test missing provider (should default to "openai")
		client, err := factory.CreateClient(ClientConfig{
			Model:  "gpt-4",
			APIKey: "test-key",
		})
		if err != nil {
			t.Errorf("Expected no error for missing provider (should default to openai), got: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}

		// Test missing model
		_, err = factory.CreateClient(ClientConfig{Provider: "openai"})
		if err == nil {
			t.Error("Expected error for missing model")
		}
	})

	t.Run("Factory Creates Correct Clients", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()

		testCases := []struct {
			name     string
			config   ClientConfig
			expected string
			hasError bool
		}{
			{
				name: "OpenAI client",
				config: ClientConfig{
					Provider: "openai",
					Model:    "gpt-4",
					APIKey:   "test-key",
				},
				expected: "openai",
				hasError: false,
			},
			{
				name: "Gemini client",
				config: ClientConfig{
					Provider: "gemini",
					Model:    "gemini-1.5-flash",
					APIKey:   "test-key",
				},
				expected: "gemini",
				hasError: false,
			},
			{
				name: "Ollama client",
				config: ClientConfig{
					Provider: "ollama",
					Model:    "llama2",
				},
				expected: "ollama",
				hasError: false,
			},
			{
				name: "OpenRouter client",
				config: ClientConfig{
					Provider: "openrouter",
					Model:    "openai/gpt-4",
					APIKey:   "sk-or-test-key",
				},
				expected: "openrouter",
				hasError: false,
			},
			{
				name: "Unsupported provider",
				config: ClientConfig{
					Provider: "unsupported",
					Model:    "some-model",
				},
				hasError: true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				client, err := factory.CreateClient(tc.config)

				if tc.hasError {
					if err == nil {
						t.Error("Expected error for unsupported provider")
					}
					return
				}

				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				if client == nil {
					t.Error("Expected client to be created")
					return
				}

				modelInfo := client.GetModelInfo()
				if modelInfo.Provider != tc.expected {
					t.Errorf("Expected provider '%s', got '%s'", tc.expected, modelInfo.Provider)
				}

				_ = client.Close()
			})
		}
	})

	t.Run("OpenRouter Factory Integration", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()

		t.Run("Valid OpenRouter Configuration", func(t *testing.T) {
			t.Parallel()
			config := ClientConfig{
				Provider: "openrouter",
				Model:    "openai/gpt-4",
				APIKey:   "sk-or-test-key",
				BaseURL:  "https://openrouter.ai/api/v1",
				Extra: map[string]string{
					"site_url": "https://myapp.com",
					"app_name": "TestApp",
				},
			}

			client, err := factory.CreateClient(config)
			if err != nil {
				t.Errorf("Expected no error for valid OpenRouter config, got: %v", err)
				return
			}

			if client == nil {
				t.Error("Expected client to be created")
				return
			}

			modelInfo := client.GetModelInfo()
			if modelInfo.Provider != "openrouter" {
				t.Errorf("Expected provider 'openrouter', got '%s'", modelInfo.Provider)
			}

			if modelInfo.Name != "openai/gpt-4" {
				t.Errorf("Expected model 'openai/gpt-4', got '%s'", modelInfo.Name)
			}

			_ = client.Close()
		})

		t.Run("OpenRouter Missing API Key", func(t *testing.T) {
			t.Parallel()
			config := ClientConfig{
				Provider: "openrouter",
				Model:    "openai/gpt-4",
				// APIKey is missing
			}

			client, err := factory.CreateClient(config)
			if err == nil {
				if client != nil {
					_ = client.Close()
				}
				t.Error("Expected error for missing API key")
				return
			}

			// Verify it's the correct error type
			if llmErr, ok := err.(*Error); ok {
				if llmErr.Code != "missing_api_key" {
					t.Errorf("Expected error code 'missing_api_key', got '%s'", llmErr.Code)
				}
				if llmErr.Type != "authentication_error" {
					t.Errorf("Expected error type 'authentication_error', got '%s'", llmErr.Type)
				}
			} else {
				t.Errorf("Expected *Error type, got %T", err)
			}
		})

		t.Run("OpenRouter Case Insensitive", func(t *testing.T) {
			t.Parallel()
			config := ClientConfig{
				Provider: "OPENROUTER", // Test case insensitivity
				Model:    "openai/gpt-3.5-turbo",
				APIKey:   "sk-or-test-key",
			}

			client, err := factory.CreateClient(config)
			if err != nil {
				t.Errorf("Expected no error for case insensitive provider, got: %v", err)
				return
			}

			if client == nil {
				t.Error("Expected client to be created")
				return
			}

			modelInfo := client.GetModelInfo()
			if modelInfo.Provider != "openrouter" {
				t.Errorf("Expected provider 'openrouter', got '%s'", modelInfo.Provider)
			}

			_ = client.Close()
		})
	})

	// Removed tests for deleted factory functions (CreateClientFromModelConfig and ValidateModelConfig)
	// These functions were removed as part of the interface simplification
}
