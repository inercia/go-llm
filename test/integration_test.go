package test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/llm"
)

// TestIntegrationOverall provides a comprehensive integration test that validates
// the entire flow from environment configuration to actual LLM communication.
// Individual functionality is tested in separate files:
// - chat_test.go: Basic chat functionality
// - tools_test.go: Tool calling capabilities
// - multimodal_test.go: Vision and file processing
// - streaming_test.go: Streaming responses
// - factory_test.go: Factory and configuration
func TestIntegrationOverall(t *testing.T) {
	// Create client using environment configuration
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx := context.Background()

	t.Run("end_to_end_functionality", func(t *testing.T) {
		// Test basic chat completion
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Hello! Please respond with 'Integration test successful' if you can understand this message."),
				}},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Integration test chat should succeed")
		require.NotNil(t, resp, "Response should not be nil")
		require.Len(t, resp.Choices, 1, "Should have exactly one choice")

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Response text should not be empty")

		// Log the successful integration
		info := client.GetModelInfo()
		t.Logf("✅ Integration test successful!")
		t.Logf("   Provider: %s", info.Provider)
		t.Logf("   Model: %s", info.Name)
		t.Logf("   Response: %s", responseText)
		t.Logf("   Capabilities: Tools=%t, Vision=%t, Streaming=%t",
			info.SupportsTools, info.SupportsVision, info.SupportsStreaming)
	})

	t.Run("provider_health_check", func(t *testing.T) {
		// Verify the provider is responding properly
		info := client.GetModelInfo()

		require.NotEmpty(t, info.Provider, "Provider should be available")
		require.NotEmpty(t, info.Name, "Model should be specified")
		require.Greater(t, info.MaxTokens, 0, "MaxTokens should be positive")

		t.Logf("Provider health check passed: %s/%s", info.Provider, info.Name)
	})
}
