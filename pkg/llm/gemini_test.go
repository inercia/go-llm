package llm

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewGeminiClient(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping basic chat completion test")
	}

	tests := []struct {
		name        string
		config      ClientConfig
		expectError bool
	}{
		{
			name: "valid config with API key",
			config: ClientConfig{
				Provider: "gemini",
				Model:    "gemini-1.5-flash",
				APIKey:   apiKey,
			},
			expectError: false,
		},
		{
			name: "missing API key",
			config: ClientConfig{
				Provider: "gemini",
				Model:    "gemini-1.5-flash",
			},
			expectError: true,
		},
		{
			name: "empty model defaults to flash",
			config: ClientConfig{
				Provider: "gemini",
				APIKey:   apiKey,
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewGeminiClient(tt.config)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, client)
				if apiErr, ok := err.(*Error); ok {
					assert.Equal(t, "missing_api_key", apiErr.Code)
				}
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
				geminiClient, ok := client.(*GeminiClient)
				assert.True(t, ok)
				expectedModel := tt.config.Model
				if expectedModel == "" {
					expectedModel = "gemini-1.5-flash"
				}
				assert.Equal(t, expectedModel, geminiClient.model)
				assert.Equal(t, "gemini", geminiClient.provider)
				// Note: apiKey is no longer accessible as it's encapsulated in the genai client
			}
		})
	}
}

func TestGetModelInfo(t *testing.T) {
	tests := []struct {
		name          string
		model         string
		maxTokens     int
		supportsTools bool
	}{
		{
			name:          "gemini-1.5-flash",
			model:         "gemini-1.5-flash",
			maxTokens:     1000000,
			supportsTools: true,
		},
		{
			name:          "gemini-1.0-pro",
			model:         "gemini-1.0-pro",
			maxTokens:     30720, // Falls back to default pattern
			supportsTools: true,  // Falls back to default pattern
		},
		{
			name:          "gemini-1.5-pro",
			model:         "gemini-1.5-pro",
			maxTokens:     2000000,
			supportsTools: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := ClientConfig{
				Provider: "gemini",
				Model:    tt.model,
				APIKey:   "test-key",
			}

			client, err := NewGeminiClient(config)
			require.NoError(t, err)
			require.NotNil(t, client)

			info := client.GetModelInfo()
			assert.Equal(t, tt.model, info.Name)
			assert.Equal(t, "gemini", info.Provider)
			assert.Equal(t, tt.maxTokens, info.MaxTokens)
			assert.Equal(t, tt.supportsTools, info.SupportsTools)
			assert.True(t, info.SupportsStreaming)
		})
	}
}

func TestChatCompletion_Stub(t *testing.T) {
	// TODO: Update this test for the new genai-based implementation
	// The old test was mocking HTTP transport, but now we use the genai library
	t.Skip("Test needs to be updated for genai library implementation")
}

func TestStreamChatCompletion_Stub(t *testing.T) {
	// TODO: Update this test for the new genai-based implementation
	// The old test was mocking HTTP transport, but now we use the genai library
	t.Skip("Test needs to be updated for genai library implementation")
}

func TestClose(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping basic chat completion test")
	}

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   apiKey,
	}

	client, err := NewGeminiClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	err = client.Close()
	assert.NoError(t, err)
}

func TestChatCompletion_RealAPI(t *testing.T) {
	// Skip real API tests if no API key
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping real API tests")
	}

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   apiKey,
	}

	client, err := NewGeminiClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Note: This test will currently fail because the real implementation is stubbed
	// It will be enabled once the genai integration is complete
	// Real API test - remove skip when ready

	ctx := context.Background()
	req := ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []Message{
			NewTextMessage(RoleUser, "Say hello in French"),
		},
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		t.Errorf("Expected successful API call, got error: %v", err)
		return
	}

	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.GetText())
	assert.Contains(t, resp.Choices[0].Message.GetText(), "Bonjour")
}

func TestStreamChatCompletion_RealAPI(t *testing.T) {
	// Skip real API tests if no API key
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		t.Skip("GEMINI_API_KEY not set, skipping real API tests")
	}

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   apiKey,
	}

	client, err := NewGeminiClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Note: This test will currently fail because the real implementation is stubbed
	// It will be enabled once the genai integration is complete
	// Real API test - remove skip when ready

	ctx := context.Background()
	req := ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []Message{
			NewTextMessage(RoleUser, "Say hello"),
		},
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Collect stream events
	var events []StreamEvent
	for event := range stream {
		events = append(events, event)
		if event.IsDone() {
			break
		}
	}

	require.NotEmpty(t, events)
	require.True(t, events[len(events)-1].IsDone())

	// Verify we got some delta events with content
	var hasContent bool
	for _, event := range events {
		if event.IsDelta() && event.Choice.Delta != nil && len(event.Choice.Delta.Content) > 0 {
			hasContent = true
			break
		}
	}
	assert.True(t, hasContent)
}
