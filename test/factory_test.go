package test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/factory"
	"github.com/inercia/go-llm/pkg/llm"
)

func TestFactoryWithEnvironmentConfig(t *testing.T) {
	t.Run("get_llm_from_env", func(t *testing.T) {
		// Test that GetLLMFromEnv returns a valid configuration
		config := llm.GetLLMFromEnv()

		assert.NotEmpty(t, config.Provider, "Provider should not be empty")
		assert.NotEmpty(t, config.Model, "Model should not be empty")
		assert.Greater(t, config.Timeout, time.Duration(0), "Timeout should be positive")

		t.Logf("Environment config: Provider=%s, Model=%s, Timeout=%v",
			config.Provider, config.Model, config.Timeout)

		// Test that we can create a client with this config
		f := factory.New()
		client, err := f.CreateClient(config)
		require.NoError(t, err, "Should be able to create client from environment config")
		require.NotNil(t, client, "Client should not be nil")

		// Verify client info matches config
		info := client.GetModelInfo()
		assert.Equal(t, config.Provider, info.Provider, "Client provider should match config")
		assert.Equal(t, config.Model, info.Name, "Client model should match config")

		_ = client.Close()
	})

	t.Run("factory_basic_functionality", func(t *testing.T) {
		factory := factory.New()
		require.NotNil(t, factory, "Factory should not be nil")

		// Test creating client with environment config
		config := llm.GetLLMFromEnv()
		client, err := factory.CreateClient(config)
		require.NoError(t, err, "Factory should create client successfully")
		require.NotNil(t, client, "Created client should not be nil")

		// Test client functionality
		info := client.GetModelInfo()
		assert.NotEmpty(t, info.Provider, "Client should have provider")
		assert.NotEmpty(t, info.Name, "Client should have model name")
		assert.Greater(t, info.MaxTokens, 0, "Client should have positive max tokens")

		t.Logf("Created client: %s/%s (MaxTokens: %d, Tools: %t, Vision: %t)",
			info.Provider, info.Name, info.MaxTokens, info.SupportsTools, info.SupportsVision)

		// Test close
		err = client.Close()
		assert.NoError(t, err, "Client close should not error")
	})
}

func TestFactoryCustomConfigurations(t *testing.T) {
	factory := factory.New()

	t.Run("custom_timeout", func(t *testing.T) {
		config := llm.GetLLMFromEnv()
		config.Timeout = 5 * time.Second // Custom timeout

		client, err := factory.CreateClient(config)
		require.NoError(t, err, "Should create client with custom timeout")
		require.NotNil(t, client)

		// Unfortunately we can't easily test that the timeout is actually applied
		// without making a request, but we can verify the client was created
		info := client.GetModelInfo()
		assert.NotEmpty(t, info.Provider, "Client should be functional")

		_ = client.Close()
	})

	t.Run("custom_base_url", func(t *testing.T) {
		config := llm.GetLLMFromEnv()

		// Only test custom base URL if we're using a provider that supports it
		if config.Provider == "openai" || config.Provider == "ollama" {
			switch config.Provider {
			case "openai":
				config.BaseURL = "https://api.openai.com/v1" // Standard URL
			case "ollama":
				config.BaseURL = "http://localhost:11434" // Standard Ollama URL
			}

			client, err := factory.CreateClient(config)
			if err != nil {
				t.Logf("Custom base URL failed (may be expected): %v", err)
			} else {
				require.NotNil(t, client)
				t.Logf("Custom base URL successful for %s", config.Provider)
				_ = client.Close()
			}
		} else {
			t.Skipf("Custom base URL test not applicable for provider: %s", config.Provider)
		}
	})

	t.Run("max_retries", func(t *testing.T) {
		config := llm.GetLLMFromEnv()
		config.MaxRetries = 3

		client, err := factory.CreateClient(config)
		require.NoError(t, err, "Should create client with max retries")
		require.NotNil(t, client)

		info := client.GetModelInfo()
		assert.NotEmpty(t, info.Provider, "Client should be functional")

		_ = client.Close()
	})
}

func TestFactoryErrorHandling(t *testing.T) {
	factory := factory.New()

	t.Run("invalid_provider", func(t *testing.T) {
		config := llm.ClientConfig{
			Provider: "nonexistent-provider",
			Model:    "some-model",
			APIKey:   "test-key",
		}

		_, err := factory.CreateClient(config)
		assert.Error(t, err, "Should error for invalid provider")

		// Should be a proper LLM error
		if llmErr, ok := err.(*llm.Error); ok {
			assert.NotEmpty(t, llmErr.Message, "Error should have message")
			assert.Contains(t, llmErr.Code, "provider", "Error code should mention provider")
		}
	})

	t.Run("missing_model", func(t *testing.T) {
		config := llm.ClientConfig{
			Provider: "openai",
			// Missing model
			APIKey: "test-key",
		}

		_, err := factory.CreateClient(config)
		assert.Error(t, err, "Should error for missing model")

		if llmErr, ok := err.(*llm.Error); ok {
			assert.NotEmpty(t, llmErr.Message, "Error should have message")
		}
	})

	t.Run("empty_config", func(t *testing.T) {
		config := llm.ClientConfig{}

		_, err := factory.CreateClient(config)
		assert.Error(t, err, "Should error for empty config")
	})
}

func TestFactoryProviderCapabilities(t *testing.T) {
	factory := factory.New()
	config := llm.GetLLMFromEnv()

	client, err := factory.CreateClient(config)
	require.NoError(t, err, "Should create client")
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	info := client.GetModelInfo()

	t.Run("model_info_validation", func(t *testing.T) {
		assert.NotEmpty(t, info.Provider, "Provider should not be empty")
		assert.NotEmpty(t, info.Name, "Model name should not be empty")
		assert.Greater(t, info.MaxTokens, 0, "Max tokens should be positive")

		// Log capabilities for debugging
		t.Logf("Provider capabilities:")
		t.Logf("  Provider: %s", info.Provider)
		t.Logf("  Model: %s", info.Name)
		t.Logf("  Max Tokens: %d", info.MaxTokens)
		t.Logf("  Supports Tools: %t", info.SupportsTools)
		t.Logf("  Supports Vision: %t", info.SupportsVision)
		t.Logf("  Supports Streaming: %t", info.SupportsStreaming)
		t.Logf("  Supports Files: %t", info.SupportsFiles)
	})

	t.Run("provider_specific_validation", func(t *testing.T) {
		switch info.Provider {
		case "openai":
			// OpenAI should support streaming, but tools/vision depend on the specific model
			assert.True(t, info.SupportsStreaming, "OpenAI should support streaming")

			// Log the actual capabilities instead of assuming
			t.Logf("OpenAI model %s - Tools: %t, Vision: %t",
				info.Name, info.SupportsTools, info.SupportsVision)

		case "gemini":
			// Gemini capabilities
			assert.True(t, info.SupportsStreaming, "Gemini should support streaming")

		case "ollama":
			// Ollama capabilities depend on model
			// Basic validation
			assert.Greater(t, info.MaxTokens, 0, "Ollama should have positive max tokens")

		case "mock":
			// Mock provider supports everything for testing
			assert.True(t, info.SupportsTools, "Mock should support tools")
			assert.True(t, info.SupportsVision, "Mock should support vision")
			assert.True(t, info.SupportsStreaming, "Mock should support streaming")

		default:
			t.Logf("Unknown provider %s, skipping specific validation", info.Provider)
		}
	})
}

func TestFactoryMultipleClients(t *testing.T) {
	factory := factory.New()
	config := llm.GetLLMFromEnv()

	t.Run("create_multiple_clients", func(t *testing.T) {
		// Create multiple clients with same config
		clients := make([]llm.Client, 3)

		for i := 0; i < 3; i++ {
			client, err := factory.CreateClient(config)
			require.NoError(t, err, "Should create client %d", i)
			require.NotNil(t, client, "Client %d should not be nil", i)
			clients[i] = client
		}

		// Verify all clients work
		for i, client := range clients {
			info := client.GetModelInfo()
			assert.Equal(t, config.Provider, info.Provider, "Client %d should have correct provider", i)
			assert.Equal(t, config.Model, info.Name, "Client %d should have correct model", i)
		}

		// Clean up
		for i, client := range clients {
			err := client.Close()
			assert.NoError(t, err, "Client %d should close without error", i)
		}

		t.Logf("Successfully created and closed %d clients", len(clients))
	})

	t.Run("different_configurations", func(t *testing.T) {
		baseConfig := llm.GetLLMFromEnv()

		// Create configs with different timeouts
		configs := []llm.ClientConfig{
			{
				Provider: baseConfig.Provider,
				Model:    baseConfig.Model,
				APIKey:   baseConfig.APIKey,
				BaseURL:  baseConfig.BaseURL,
				Timeout:  5 * time.Second,
			},
			{
				Provider: baseConfig.Provider,
				Model:    baseConfig.Model,
				APIKey:   baseConfig.APIKey,
				BaseURL:  baseConfig.BaseURL,
				Timeout:  10 * time.Second,
			},
		}

		clients := make([]llm.Client, len(configs))

		for i, cfg := range configs {
			client, err := factory.CreateClient(cfg)
			require.NoError(t, err, "Should create client with config %d", i)
			require.NotNil(t, client)
			clients[i] = client
		}

		// Clean up
		for _, client := range clients {
			_ = client.Close()
		}

		t.Logf("Successfully created clients with %d different configurations", len(configs))
	})
}
