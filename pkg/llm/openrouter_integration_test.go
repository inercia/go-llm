package llm

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOpenRouterBasicChatCompletionIntegration tests basic chat completion with real OpenRouter API
func TestOpenRouterBasicChatCompletionIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Get test model from environment or use default
	testModel := os.Getenv("OPENROUTER_TEST_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-3.5-turbo"
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "Hello! Please respond with a short greeting."),
		},
		MaxTokens: intPtr(50), // Keep response short for faster tests
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
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

	// Get test model from environment or use default
	testModel := os.Getenv("OPENROUTER_TEST_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-3.5-turbo"
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "Count from 1 to 5, one number per line."),
		},
		MaxTokens: intPtr(50), // Keep response short for faster tests
		Stream:    true,
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	events := consumeStream(t, stream)

	// Verify we got events
	require.Greater(t, len(events), 0)

	// Check for error events
	hasError := false
	hasDelta := false
	hasDone := false
	var fullContent string

	for _, event := range events {
		if event.IsError() {
			hasError = true
			t.Logf("Error event: %v", event.Error)
		} else if event.Type == "delta" && event.Choice != nil && event.Choice.Delta != nil {
			hasDelta = true
			if len(event.Choice.Delta.Content) > 0 {
				if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
					fullContent += textContent.GetText()
				}
			}
		} else if event.IsDone() {
			hasDone = true
		}
	}

	require.False(t, hasError, "Stream should not contain error events")
	assert.True(t, hasDelta, "Stream should contain delta events")
	assert.True(t, hasDone, "Stream should contain done event")
	assert.NotEmpty(t, fullContent, "Stream should produce content")
}

// TestOpenRouterMultiModalIntegration tests multi-modal content with vision-capable models
func TestOpenRouterMultiModalIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Use a vision-capable model for this test
	testModel := os.Getenv("OPENROUTER_VISION_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-4o-mini" // Default vision-capable model
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports vision
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsVision {
		t.Skipf("Model %s does not support vision, skipping multi-modal test", testModel)
	}

	ctx := context.Background()

	// Test with fixture images
	fixtureImages := getFixtureImages(t)
	if len(fixtureImages) == 0 {
		t.Skip("No fixture images available for multi-modal testing")
	}

	// Test with the first available image
	imageName := fixtureImages[0]
	data, mimeType := loadFixtureImage(t, imageName)

	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("What do you see in this image? Please provide a brief description."),
			NewImageContentFromBytes(data, mimeType),
		},
	}

	req := ChatRequest{
		Model:     testModel,
		Messages:  []Message{message},
		MaxTokens: intPtr(100), // Keep response reasonable
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.GetText())

	// Verify the response mentions visual content
	responseText := resp.Choices[0].Message.GetText()
	assert.NotEmpty(t, responseText)
	t.Logf("Vision response for %s: %s", imageName, responseText)
}

// TestOpenRouterMultiModalStreamingIntegration tests streaming multi-modal content
func TestOpenRouterMultiModalStreamingIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Use a vision-capable model for this test
	testModel := os.Getenv("OPENROUTER_VISION_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-4o-mini" // Default vision-capable model
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Check if the model supports vision
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsVision {
		t.Skipf("Model %s does not support vision, skipping multi-modal streaming test", testModel)
	}

	ctx := context.Background()

	// Test with fixture images
	fixtureImages := getFixtureImages(t)
	if len(fixtureImages) == 0 {
		t.Skip("No fixture images available for multi-modal testing")
	}

	// Test with the first available image
	imageName := fixtureImages[0]
	data, mimeType := loadFixtureImage(t, imageName)

	message := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Describe this image briefly."),
			NewImageContentFromBytes(data, mimeType),
		},
	}

	req := ChatRequest{
		Model:     testModel,
		Messages:  []Message{message},
		MaxTokens: intPtr(100), // Keep response reasonable
		Stream:    true,
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	events := consumeStream(t, stream)

	// Verify we got events
	require.Greater(t, len(events), 0)

	// Check for error events
	hasError := false
	hasDelta := false
	hasDone := false
	var fullContent string

	for _, event := range events {
		if event.IsError() {
			hasError = true
			t.Logf("Error event: %v", event.Error)
		} else if event.Type == "delta" && event.Choice != nil && event.Choice.Delta != nil {
			hasDelta = true
			if len(event.Choice.Delta.Content) > 0 {
				if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
					fullContent += textContent.GetText()
				}
			}
		} else if event.IsDone() {
			hasDone = true
		}
	}

	require.False(t, hasError, "Stream should not contain error events")
	assert.True(t, hasDelta, "Stream should contain delta events")
	assert.True(t, hasDone, "Stream should contain done event")
	assert.NotEmpty(t, fullContent, "Stream should produce content")
	t.Logf("Vision streaming response for %s: %s", imageName, fullContent)
}

// TestOpenRouterToolUsageIntegration tests tool usage with compatible models
func TestOpenRouterToolUsageIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	// Use a tool-capable model for this test
	testModel := os.Getenv("OPENROUTER_TOOL_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-4o-mini" // Default tool-capable model
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
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
	weatherTool := Tool{
		Type: "function",
		Function: ToolFunction{
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

	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "What's the weather like in New York?"),
		},
		Tools:     []Tool{weatherTool},
		MaxTokens: intPtr(150),
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

	// Use a tool-capable model for this test
	testModel := os.Getenv("OPENROUTER_TOOL_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-4o-mini" // Default tool-capable model
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	// Set custom base URL if provided
	if baseURL := os.Getenv("OPENROUTER_BASE_URL"); baseURL != "" {
		config.BaseURL = baseURL
	}

	client, err := NewOpenRouterClient(config)
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
	calcTool := Tool{
		Type: "function",
		Function: ToolFunction{
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

	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "What is 15 + 27?"),
		},
		Tools:     []Tool{calcTool},
		MaxTokens: intPtr(150),
		Stream:    true,
	}

	stream, err := client.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	events := consumeStream(t, stream)

	// Verify we got events
	require.Greater(t, len(events), 0)

	// Check for error events and tool calls
	hasError := false
	hasDelta := false
	hasDone := false
	hasToolCall := false
	var fullContent string

	for _, event := range events {
		if event.IsError() {
			hasError = true
			t.Logf("Error event: %v", event.Error)
		} else if event.Type == "delta" && event.Choice != nil && event.Choice.Delta != nil {
			hasDelta = true

			// Check for content
			if len(event.Choice.Delta.Content) > 0 {
				if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
					fullContent += textContent.GetText()
				}
			}

			// Check for tool calls
			if len(event.Choice.Delta.ToolCalls) > 0 {
				hasToolCall = true
				t.Logf("Tool call delta received: %+v", event.Choice.Delta.ToolCalls[0])
			}
		} else if event.IsDone() {
			hasDone = true
			if event.Choice != nil && event.Choice.FinishReason == "tool_calls" {
				hasToolCall = true
			}
		}
	}

	require.False(t, hasError, "Stream should not contain error events")
	assert.True(t, hasDelta, "Stream should contain delta events")
	assert.True(t, hasDone, "Stream should contain done event")

	// The model should either call the tool or respond normally
	if hasToolCall {
		t.Log("Model made tool call in streaming mode")
	} else {
		assert.NotEmpty(t, fullContent, "Stream should produce content if no tool call")
		t.Logf("Model responded without tool call: %s", fullContent)
	}
}

// TestOpenRouterErrorHandlingIntegration tests error handling with invalid requests
func TestOpenRouterErrorHandlingIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY not set, skipping real integration test")
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    "invalid/model-that-does-not-exist",
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	req := ChatRequest{
		Model: "invalid/model-that-does-not-exist",
		Messages: []Message{
			NewTextMessage(RoleUser, "This should fail"),
		},
	}

	// Test non-streaming error
	resp, err := client.ChatCompletion(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)

	// Verify error structure
	if llmErr, ok := err.(*Error); ok {
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
		if llmErr, ok := err.(*Error); ok {
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

	factory := NewFactory()

	// Get test model from environment or use default
	testModel := os.Getenv("OPENROUTER_TEST_MODEL")
	if testModel == "" {
		testModel = "openai/gpt-3.5-turbo"
	}

	config := ClientConfig{
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

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	// Verify it's an OpenRouter client
	modelInfo := client.GetModelInfo()
	assert.Equal(t, "openrouter", modelInfo.Provider)
	assert.Equal(t, testModel, modelInfo.Name)

	// Test basic functionality
	ctx := context.Background()
	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "Hello from factory integration test!"),
		},
		MaxTokens: intPtr(30),
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
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
		testModel = "openai/gpt-3.5-turbo"
	}

	config := ClientConfig{
		Provider: "openrouter",
		Model:    testModel,
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
	}

	client, err := NewOpenRouterClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Verify client is initially set
	assert.NotNil(t, client.client, "Client should not be nil initially")

	// Make a quick request to ensure client is working
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := ChatRequest{
		Model: testModel,
		Messages: []Message{
			NewTextMessage(RoleUser, "Hi"),
		},
		MaxTokens: intPtr(5), // Very short response
	}

	resp, err := client.ChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, resp)

	// Now test Close method
	err = client.Close()
	require.NoError(t, err, "Close() should not return error")

	// Verify client is cleaned up
	assert.Nil(t, client.client, "Client should be nil after Close()")

	// Test that Close can be called multiple times safely
	err = client.Close()
	require.NoError(t, err, "Second Close() should not return error")
}

// Helper functions are imported from other test files
