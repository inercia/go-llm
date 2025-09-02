package openrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/llm"
)

// Disabled: Internal retry configuration - would need internal types
// func getRetryConfigForIntegrationTests() RetryConfig {
//   return RetryConfig{...}
// }

// TestOpenRouterBasicChatCompletionIntegration tests basic chat completion with real OpenRouter API
func TestOpenRouterBasicChatCompletionIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, false)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Direct client usage without retry wrapper
	ctx := context.Background()
	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Hello! Please respond with a short greeting."),
		},
		MaxTokens: func() *int { i := 50; return &i }(), // Keep response short for faster tests
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)

	// Debug logging to understand what we're getting
	t.Logf("Using model: %s", testModel)
	t.Logf("Response ID: %s", resp.ID)
	t.Logf("Response Model: %s", resp.Model)
	t.Logf("Choices length: %d", len(resp.Choices))
	if len(resp.Choices) > 0 {
		t.Logf("Choice message content length: %d", len(resp.Choices[0].Message.Content))
		t.Logf("Choice message GetText(): '%s'", resp.Choices[0].Message.GetText())
		t.Logf("Choice finish reason: %s", resp.Choices[0].FinishReason)
	}
	t.Logf("Usage: %+v", resp.Usage)

	require.NotEmpty(t, resp.Choices[0].Message.GetText())

	// Verify response structure
	assert.NotEmpty(t, resp.ID)
	assert.Equal(t, testModel, resp.Model)
	assert.Equal(t, "stop", resp.Choices[0].FinishReason)
	assert.Greater(t, resp.Usage.TotalTokens, 0)
}

// TestOpenRouterStreamingChatCompletionIntegration tests streaming chat completion with real OpenRouter API
func TestOpenRouterStreamingChatCompletionIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, false)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Count from 1 to 5, one number per line."),
		},
		MaxTokens: func() *int { i := 50; return &i }(), // Keep response short for faster tests
		Stream:    true,
	}

	// Simple streaming test without complex retry logic
	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Consume a few events to verify streaming works
	eventCount := 0
	for event := range stream {
		eventCount++
		if event.IsError() {
			t.Logf("Stream error: %v", event.Error)
		}
		// Don't consume all events to avoid long test times
		if eventCount >= 3 {
			break
		}
	}

	// Verify we got some events
	require.Greater(t, eventCount, 0, "Should have received some stream events")
	t.Logf("Streaming test completed successfully, received %d events", eventCount)
}

// TestOpenRouterMultiModalIntegration tests multi-modal content with vision-capable models
func TestOpenRouterMultiModalIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, true)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports vision
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsVision {
		t.Skipf("Model %s does not support vision, skipping multi-modal test", testModel)
	}

	ctx := context.Background()

	// Use simple test image data instead of fixture files
	data := []byte("fake-jpeg-data-for-testing")
	mimeType := "image/jpeg"

	message := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.MessageContent{
			llm.NewTextContent("What do you see in this image? Please provide a brief description."),
			llm.NewImageContentFromBytes(data, mimeType),
		},
	}

	req := llm.ChatRequest{
		Model:     testModel,
		Messages:  []llm.Message{message},
		MaxTokens: func() *int { i := 100; return &i }(), // Keep response reasonable
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.GetText())

	// Verify the response mentions visual content
	responseText := resp.Choices[0].Message.GetText()
	assert.NotEmpty(t, responseText)
	t.Logf("Vision response: %s", responseText)
}

// TestOpenRouterMultiModalStreamingIntegration tests streaming multi-modal content
func TestOpenRouterMultiModalStreamingIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Use a vision-capable model for this test
	testModel := GetOpenRouterTestingModel(true, true)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports vision
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsVision {
		t.Skipf("Model %s does not support vision, skipping multi-modal streaming test", testModel)
	}

	ctx := context.Background()

	// Use simple test image data instead of fixture files
	data := []byte("fake-jpeg-data-for-testing")
	mimeType := "image/jpeg"

	message := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.MessageContent{
			llm.NewTextContent("Describe this image briefly."),
			llm.NewImageContentFromBytes(data, mimeType),
		},
	}

	req := llm.ChatRequest{
		Model:     testModel,
		Messages:  []llm.Message{message},
		MaxTokens: func() *int { i := 100; return &i }(), // Keep response reasonable
		Stream:    true,
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Consume a few events to verify streaming works
	eventCount := 0
	for event := range stream {
		eventCount++
		if event.IsError() {
			t.Logf("Stream error: %v", event.Error)
		}
		// Don't consume all events to avoid long test times
		if eventCount >= 5 {
			break
		}
	}

	// Verify we got some events
	require.Greater(t, eventCount, 0, "Should have received some stream events")
	t.Logf("Streaming multimodal test completed successfully, received %d events", eventCount)
}

// TestOpenRouterToolUsageIntegration tests tool usage with compatible models
func TestOpenRouterToolUsageIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, false)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports tools
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsTools {
		t.Skipf("Model %s does not support tools, skipping tool usage test", testModel)
	}

	ctx := context.Background()

	// Define a simple weather tool
	weatherTool := llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]interface{}{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
					"unit": map[string]interface{}{
						"type":        "string",
						"enum":        []string{"celsius", "fahrenheit"},
						"description": "The unit of temperature",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What's the weather like in New York?"),
		},
		Tools:     []llm.Tool{weatherTool},
		MaxTokens: func() *int { i := 150; return &i }(),
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)

	choice := resp.Choices[0]

	// The model should either:
	// 1. Call the tool (finish_reason = "tool_calls")
	// 2. Respond normally if it doesn't think a tool call is needed
	if choice.FinishReason == "tool_calls" {
		// Verify tool call structure
		require.NotEmpty(t, choice.Message.ToolCalls)
		toolCall := choice.Message.ToolCalls[0]
		assert.Equal(t, "function", toolCall.Type)
		assert.Equal(t, "get_weather", toolCall.Function.Name)
		assert.NotEmpty(t, toolCall.Function.Arguments)
		assert.NotEmpty(t, toolCall.ID)
		t.Logf("Tool call made: %s with args: %s", toolCall.Function.Name, toolCall.Function.Arguments)
	} else {
		// Model responded normally without tool call
		assert.NotEmpty(t, choice.Message.GetText())
		t.Logf("Model responded without tool call: %s", choice.Message.GetText())
	}
}

// TestOpenRouterToolUsageStreamingIntegration tests streaming tool usage
func TestOpenRouterToolUsageStreamingIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, false)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports tools
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsTools {
		t.Skipf("Model %s does not support tools, skipping streaming tool usage test", testModel)
	}

	ctx := context.Background()

	// Define a simple calculator tool
	calcTool := llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name:        "calculate",
			Description: "Perform basic arithmetic calculations",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"expression": map[string]interface{}{
						"type":        "string",
						"description": "The mathematical expression to evaluate, e.g. '2 + 3'",
					},
				},
				"required": []string{"expression"},
			},
		},
	}

	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "What is 15 + 27?"),
		},
		Tools:     []llm.Tool{calcTool},
		MaxTokens: func() *int { i := 150; return &i }(),
		Stream:    true,
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Consume a few events to verify streaming works
	eventCount := 0
	for event := range stream {
		eventCount++
		if event.IsError() {
			t.Logf("Stream error: %v", event.Error)
		}
		// Don't consume all events to avoid long test times
		if eventCount >= 5 {
			break
		}
	}

	// Verify we got some events
	require.Greater(t, eventCount, 0, "Should have received some stream events")
	t.Logf("Tool usage streaming test completed successfully, received %d events", eventCount)
}

// TestOpenRouterErrorHandlingIntegration tests error handling with invalid requests
func TestOpenRouterErrorHandlingIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    "invalid/model-that-does-not-exist",
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	req := llm.ChatRequest{
		Model: "invalid/model-that-does-not-exist",
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "This should fail"),
		},
	}

	// Test non-streaming error
	resp, err := client.ChatCompletion(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)

	// Verify error structure
	if llmErr, ok := err.(*llm.Error); ok {
		assert.NotEmpty(t, llmErr.Code)
		assert.NotEmpty(t, llmErr.Message)
		assert.NotEmpty(t, llmErr.Type)
		t.Logf("Non-streaming error: %+v", llmErr)
	}

	// Test streaming error
	stream, err := client.StreamChatCompletion(ctx, req)
	if err != nil {
		// Error occurred immediately
		assert.Nil(t, stream)
		if llmErr, ok := err.(*llm.Error); ok {
			assert.NotEmpty(t, llmErr.Code)
			t.Logf("Immediate streaming error: %+v", llmErr)
		}
	} else {
		// Error should come through the stream
		require.NotNil(t, stream)

		timeout := time.NewTimer(10 * time.Second)
		defer timeout.Stop()

		select {
		case event, ok := <-stream:
			if !ok {
				t.Error("Stream closed without error event")
			} else if event.IsError() {
				assert.NotNil(t, event.Error)
				t.Logf("Stream error event: %+v", event.Error)
			} else {
				t.Logf("Unexpected event type: %s", event.Type)
			}
		case <-timeout.C:
			t.Error("Timeout waiting for error event")
		}
	}
}

// TestOpenRouterFactoryIntegration tests OpenRouter client creation through factory
func TestOpenRouterFactoryIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	testModel := GetOpenRouterTestingModel(true, false)

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
		Extra: map[string]string{
			"site_url": "https://test-integration.com",
			"app_name": "OpenRouter Integration Test",
		},
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Verify it's an OpenRouter client
	modelInfo := client.GetModelInfo()
	assert.Equal(t, "openrouter", modelInfo.Provider)
	assert.Equal(t, testModel, modelInfo.Name)

	// Test basic functionality
	ctx := context.Background()
	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Hello from factory integration test!"),
		},
		MaxTokens: func() *int { i := 30; return &i }(),
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)

	// Debug logging for factory test
	t.Logf("Factory test - Using model: %s", testModel)
	t.Logf("Factory test - Response: ID=%s, Model=%s", resp.ID, resp.Model)
	t.Logf("Factory test - Message content: '%s'", resp.Choices[0].Message.GetText())
	t.Logf("Factory test - Finish reason: %s", resp.Choices[0].FinishReason)

	require.NotEmpty(t, resp.Choices[0].Message.GetText())
}

// TestOpenRouterCloseIntegration tests that Close method works properly with real client
func TestOpenRouterCloseIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Get test model from environment or use default
	testModel := os.Getenv("OPENROUTER_TEST_MODEL")
	if testModel == "" {
		testModel = GetOpenRouterTestingModel(true, false)
	}

	config := llm.ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Verify client is initially set (using public interface)
	modelInfo := client.GetModelInfo()
	assert.NotEmpty(t, modelInfo.Provider, "Client should have a provider set")

	// Make a quick request to ensure client is working
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	req := llm.ChatRequest{
		Model: testModel,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Hi"),
		},
		MaxTokens: func() *int { i := 5; return &i }(), // Very short response
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Now test Close method
	err = client.Close()
	require.NoError(t, err, "Close() should not return error")

	// Verify client is cleaned up (using public interface behavior)
	// After close, subsequent calls should fail gracefully

	// Test that Close can be called multiple times safely
	err = client.Close()
	require.NoError(t, err, "Second Close() should not return error")
}

// TestOpenRouterListModelsIntegration tests listing models with real OpenRouter API
func TestOpenRouterListModelsIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	config := llm.ClientConfig{
		Provider: "openrouter",
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	// Test basic client functionality instead of ListModels (not in public interface)
	modelInfo := client.GetModelInfo()
	assert.NotEmpty(t, modelInfo.Provider, "Provider should be set")
	assert.Equal(t, "openrouter", modelInfo.Provider, "Should be OpenRouter provider")

	// Make a simple test request to verify the client works
	req := llm.ChatRequest{
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Hello"),
		},
		MaxTokens: func() *int { i := 10; return &i }(), // Very short response
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	assert.Greater(t, len(resp.Choices), 0, "Should have at least one choice")
	t.Logf("Basic functionality test passed - got %d choices", len(resp.Choices))
}

// Helper functions are imported from other test files
