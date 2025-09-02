package llm

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

// TestOpenAIProvider tests the OpenAI client with real API calls
func TestOpenAIProvider(t *testing.T) {
	t.Parallel()
	t.Run("CreateOpenAIClient", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "openai",
			APIKey:   "test-key",
			Model:    "gpt-4o-mini",
		})
		if err != nil {
			t.Fatalf("Failed to create OpenAI client: %v", err)
		}
		defer func() { _ = client.Close() }()

		// Test model info
		info := client.GetModelInfo()
		if info.Provider != "openai" {
			t.Errorf("Expected provider 'openai', got '%s'", info.Provider)
		}
		if info.Name != "gpt-4o-mini" {
			t.Errorf("Expected model 'gpt-4o-mini', got '%s'", info.Name)
		}
		if !info.SupportsTools {
			t.Error("gpt-4o-mini should support tools")
		}
	})

	t.Run("OpenAI_BasicChatCompletion", func(t *testing.T) {
		t.Parallel()

		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			t.Skip("OPENAI_API_KEY not set, skipping basic chat completion test")
		}

		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "openai",
			APIKey:   apiKey,
			Model:    "gpt-4o-mini",
		})
		if err != nil {
			t.Fatalf("Failed to create OpenAI client: %v", err)
		}
		defer func() { _ = client.Close() }()

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		req := ChatRequest{
			Model: "gpt-4o-mini",
			Messages: []Message{
				NewTextMessage(RoleSystem, "You are a helpful assistant. Respond with exactly 'Hello World' and nothing else."),
				NewTextMessage(RoleUser, "Say hello"),
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Fatalf("Chat completion failed: %v", err)
		}

		if resp == nil {
			t.Fatal("Response is nil")
		}
		if len(resp.Choices) == 0 {
			t.Fatal("No choices in response")
		}
		if resp.Choices[0].Message.GetText() == "" {
			t.Error("Response content is empty")
		}
		if resp.Usage.TotalTokens == 0 {
			t.Error("Usage tokens should be greater than 0")
		}

		t.Logf("✓ OpenAI response: %s (tokens: %d)",
			resp.Choices[0].Message.GetText(), resp.Usage.TotalTokens)
	})

	t.Run("OpenAI_InvalidRequest", func(t *testing.T) {
		t.Parallel()

		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			t.Skip("OPENAI_API_KEY not set, skipping invalid request test")
		}

		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "openai",
			APIKey:   apiKey,
			Model:    "gpt-4o-mini",
		})
		if err != nil {
			t.Fatalf("Failed to create OpenAI client: %v", err)
		}
		defer func() { _ = client.Close() }()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		// Test with empty messages - should return an error
		req := ChatRequest{
			Model:    "gpt-4o-mini",
			Messages: []Message{}, // Empty messages should cause error
		}

		_, err = client.ChatCompletion(ctx, req)
		if err == nil {
			t.Error("Expected error for empty messages, but got none")
		}

		// Check that it's our standardized error format
		if llmErr, ok := err.(*Error); ok {
			t.Logf("✓ OpenAI error handling: Code=%s, Type=%s", llmErr.Code, llmErr.Type)
		} else {
			t.Errorf("Expected *Error type, got %T", err)
		}
	})

	t.Run("OpenAI_MissingAPIKey", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()
		_, err := factory.CreateClient(ClientConfig{
			Provider: "openai",
			APIKey:   "", // Empty API key
			Model:    "gpt-4o-mini",
		})
		if err == nil {
			t.Error("Expected error for missing API key")
		}
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code != "missing_api_key" {
				t.Errorf("Expected code 'missing_api_key', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("OpenAI_CustomBaseURL", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "openai",
			Model:    "gpt-4o-mini",
			APIKey:   "test-key",
			BaseURL:  "https://custom-endpoint.example.com",
		}

		factory := NewFactory()
		client, err := factory.CreateClient(config)
		if err != nil {
			t.Errorf("Should be able to create client with custom base URL: %v", err)
		}
		if client != nil {
			_ = client.Close()
		}
	})
}

// TestOpenAISystemRoles tests OpenAI's native system role support
func TestOpenAISystemRoles(t *testing.T) {
	t.Parallel()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping OpenAI system role test")
	}

	factory := NewFactory()
	client, err := factory.CreateClient(ClientConfig{
		Provider: "openai",
		APIKey:   apiKey,
		Model:    "gpt-4o-mini",
	})
	if err != nil {
		t.Fatalf("Failed to create OpenAI client: %v", err)
	}
	defer func() { _ = client.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test with system role (should work natively)
	req := ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []Message{
			NewTextMessage(RoleSystem, "You are a helpful assistant that responds with exactly 'System role working' and nothing else."),
			NewTextMessage(RoleUser, "Test system role"),
		},
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		t.Fatalf("Chat completion with system role failed: %v", err)
	}

	if resp == nil || len(resp.Choices) == 0 {
		t.Fatal("Expected valid response with choices")
	}

	content := resp.Choices[0].Message.GetText()
	if !strings.Contains(strings.ToLower(content), "system") {
		t.Errorf("Expected response to acknowledge system role, got: %s", content)
	}

	t.Logf("✓ OpenAI system role working: %s", content)
}
