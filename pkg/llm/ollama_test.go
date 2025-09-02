package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

const DefaultModel = "qwen3:4b-thinking-2507-q4_K_M"

func GetDefaultOllamaModel() string {
	m := os.Getenv("OLLAMA_MODEL")
	if m == "" {
		m = DefaultModel
	}
	return m
}

// Helper function to check if Ollama is available
func isOllamaAvailable() bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://localhost:11434/api/tags")
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()
	return resp.StatusCode == http.StatusOK
}

// Helper function to check if a specific model is available in Ollama
func isOllamaModelAvailable(model string) bool {
	if !isOllamaAvailable() {
		return false
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get("http://localhost:11434/api/tags")
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return false
	}

	var result struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false
	}

	for _, m := range result.Models {
		if m.Name == model {
			return true
		}
	}
	return false
}

// TestOllamaProvider tests the Ollama client with local server
func TestOllamaProvider(t *testing.T) {
	t.Parallel()

	// Check if Ollama server is running locally
	if !isOllamaAvailable() {
		t.Skip("Ollama server not available on localhost:11434, skipping Ollama provider test")
	}

	defaultModel := GetDefaultOllamaModel()
	if !isOllamaModelAvailable(defaultModel) {
		t.Skipf("Model %s not available on this Ollama server, skipping tests", defaultModel)
	}

	t.Run("CreateOllamaClient", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "ollama",
			Model:    defaultModel,
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}
		defer func() { _ = client.Close() }()

		// Test model info
		info := client.GetModelInfo()
		if info.Provider != "ollama" {
			t.Errorf("Expected provider 'ollama', got '%s'", info.Provider)
		}
		if info.Name != defaultModel {
			t.Errorf("Expected model '%s', got '%s'", defaultModel, info.Name)
		}
		if info.SupportsTools {
			t.Error("Ollama should not support tools by default")
		}
	})

	t.Run("Ollama_BasicChatCompletion", func(t *testing.T) {
		t.Parallel()

		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "ollama",
			Model:    defaultModel,
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}
		defer func() { _ = client.Close() }()

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second) // Reduced timeout
		defer cancel()

		req := ChatRequest{
			Model: defaultModel,
			Messages: []Message{
				{Role: RoleSystem, Content: []MessageContent{NewTextContent("You are a helpful assistant. Keep responses short.")}},
				{Role: RoleUser, Content: []MessageContent{NewTextContent("Say hello in one word")}},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			// Check if it's a "model not found" error, which is common in test environments
			if strings.Contains(err.Error(), "not found") {
				t.Skipf("Model %s not available on this Ollama server: %v", defaultModel, err)
			}
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

		t.Logf("✓ Ollama response: %s", resp.Choices[0].Message.GetText())
	})

	t.Run("Ollama_InvalidModel", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()
		client, err := factory.CreateClient(ClientConfig{
			Provider: "ollama",
			Model:    "nonexistent-model",
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}
		defer func() { _ = client.Close() }()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		req := ChatRequest{
			Model: "nonexistent-model",
			Messages: []Message{
				{Role: RoleUser, Content: []MessageContent{NewTextContent("Hello")}},
			},
		}

		_, err = client.ChatCompletion(ctx, req)
		if err == nil {
			t.Error("Expected error for nonexistent model, but got none")
		}

		// Check that it's our standardized error format
		if llmErr, ok := err.(*Error); ok {
			t.Logf("✓ Ollama error handling: Code=%s, Type=%s", llmErr.Code, llmErr.Type)
		} else {
			t.Logf("Ollama error (not standardized): %T: %v", err, err)
		}
	})

	t.Run("Ollama_EmptyModel", func(t *testing.T) {
		t.Parallel()
		factory := NewFactory()
		_, err := factory.CreateClient(ClientConfig{
			Provider: "ollama",
			Model:    "", // Empty model should cause error now due to factory validation
		})
		if err == nil {
			t.Error("Expected error for empty model name")
		}
	})
}

// TestOllamaSystemRoles tests Ollama's system role handling
func TestOllamaSystemRoles(t *testing.T) {
	t.Parallel()

	if !isOllamaAvailable() {
		t.Skip("Ollama server not available on localhost:11434, skipping Ollama system role test")
	}

	defaultModel := GetDefaultOllamaModel()
	if !isOllamaModelAvailable(defaultModel) {
		t.Skipf("Model %s not available on this Ollama server, skipping system role test", defaultModel)
	}

	factory := NewFactory()
	client, err := factory.CreateClient(ClientConfig{
		Provider: "ollama",
		Model:    defaultModel,
	})
	if err != nil {
		t.Fatalf("Failed to create Ollama client: %v", err)
	}
	defer func() { _ = client.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second) // Reduced timeout
	defer cancel()

	// Test with system role (should work with Ollama)
	req := ChatRequest{
		Model: defaultModel,
		Messages: []Message{
			{Role: RoleSystem, Content: []MessageContent{NewTextContent("You are a helpful assistant. Always respond with exactly 'System role working' and nothing else.")}},
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Test system role")}},
		},
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		// Check if model is not available
		if strings.Contains(err.Error(), "not found") {
			t.Skipf("Model %s not available on this Ollama server: %v", defaultModel, err)
		}
		t.Fatalf("Chat completion with system role failed: %v", err)
	}

	if resp == nil || len(resp.Choices) == 0 {
		t.Fatal("Expected valid response with choices")
	}

	content := resp.Choices[0].Message.GetText()
	if content == "" {
		t.Error("Response content is empty")
	}

	t.Logf("✓ Ollama system role working: %s", content)
}

// TestOllamaClientUnit tests Ollama client without requiring server
func TestOllamaClientUnit(t *testing.T) {
	t.Parallel()

	t.Run("NewOllamaClient_DefaultConfig", func(t *testing.T) {
		client, err := NewOllamaClient(ClientConfig{
			Model: "test-model",
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}

		if client.model != "test-model" {
			t.Errorf("Expected model 'test-model', got '%s'", client.model)
		}
		if client.baseURL != "http://localhost:11434" {
			t.Errorf("Expected default base URL 'http://localhost:11434', got '%s'", client.baseURL)
		}
		if client.httpClient == nil {
			t.Error("HTTP client should not be nil")
		}
	})

	t.Run("NewOllamaClient_CustomConfig", func(t *testing.T) {
		client, err := NewOllamaClient(ClientConfig{
			Model:   "custom-model",
			BaseURL: "http://remote:8080",
			Timeout: 30 * time.Second,
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}

		if client.model != "custom-model" {
			t.Errorf("Expected model 'custom-model', got '%s'", client.model)
		}
		if client.baseURL != "http://remote:8080" {
			t.Errorf("Expected base URL 'http://remote:8080', got '%s'", client.baseURL)
		}
		if client.httpClient.Timeout != 30*time.Second {
			t.Errorf("Expected timeout 30s, got %v", client.httpClient.Timeout)
		}
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		client, err := NewOllamaClient(ClientConfig{
			Model: "test-model",
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}

		info := client.GetModelInfo()
		if info.Provider != "ollama" {
			t.Errorf("Expected provider 'ollama', got '%s'", info.Provider)
		}
		if info.Name != "test-model" {
			t.Errorf("Expected model name 'test-model', got '%s'", info.Name)
		}
		if info.SupportsTools {
			t.Error("Ollama should not support tools")
		}
		if info.MaxTokens != 4096 {
			t.Errorf("Expected max tokens 4096, got %d", info.MaxTokens)
		}
	})

	t.Run("Close", func(t *testing.T) {
		client, err := NewOllamaClient(ClientConfig{
			Model: "test-model",
		})
		if err != nil {
			t.Fatalf("Failed to create Ollama client: %v", err)
		}

		// Close should not return an error for Ollama client
		if err := client.Close(); err != nil {
			t.Errorf("Close should not return error, got: %v", err)
		}
	})
}
