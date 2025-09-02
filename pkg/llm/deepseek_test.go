package llm

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/cohesion-org/deepseek-go"
)

// TestNewDeepSeekClient tests the DeepSeek client constructor
func TestNewDeepSeekClient(t *testing.T) {
	t.Parallel()

	t.Run("ValidConfiguration", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
		}

		client, err := NewDeepSeekClient(config)
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		if client.provider != "deepseek" {
			t.Errorf("Expected provider 'deepseek', got '%s'", client.provider)
		}
		if client.model != "deepseek-chat" {
			t.Errorf("Expected model 'deepseek-chat', got '%s'", client.model)
		}
	})

	t.Run("MissingAPIKey", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			Model:    "deepseek-chat",
			// APIKey is intentionally missing
		}

		client, err := NewDeepSeekClient(config)
		if err == nil {
			t.Fatal("Expected error for missing API key")
		}
		if client != nil {
			t.Error("Expected nil client when API key is missing")
		}

		// Check error details
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

	t.Run("MissingModel", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			// Model is intentionally missing
		}

		client, err := NewDeepSeekClient(config)
		if err == nil {
			t.Fatal("Expected error for missing model")
		}
		if client != nil {
			t.Error("Expected nil client when model is missing")
		}

		// Check error details
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code != "missing_model" {
				t.Errorf("Expected error code 'missing_model', got '%s'", llmErr.Code)
			}
			if llmErr.Type != "validation_error" {
				t.Errorf("Expected error type 'validation_error', got '%s'", llmErr.Type)
			}
		} else {
			t.Errorf("Expected *Error type, got %T", err)
		}
	})

	t.Run("InvalidBaseURL", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
			BaseURL:  "https://", // Invalid base URL
		}

		client, err := NewDeepSeekClient(config)
		if err == nil {
			t.Fatal("Expected error for invalid base URL")
		}
		if client != nil {
			t.Error("Expected nil client when base URL is invalid")
		}

		// Check error details
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code != "invalid_base_url" {
				t.Errorf("Expected error code 'invalid_base_url', got '%s'", llmErr.Code)
			}
			if llmErr.Type != "validation_error" {
				t.Errorf("Expected error type 'validation_error', got '%s'", llmErr.Type)
			}
		} else {
			t.Errorf("Expected *Error type, got %T", err)
		}
	})

	t.Run("WithCustomBaseURL", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
			BaseURL:  "https://api.custom-deepseek.com/v1",
		}

		client, err := NewDeepSeekClient(config)
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client with custom base URL: %v", err)
		}
		defer func() { _ = client.Close() }()

		if client.config.BaseURL != config.BaseURL {
			t.Errorf("Expected base URL '%s', got '%s'", config.BaseURL, client.config.BaseURL)
		}
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		t.Parallel()
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
		}

		client, err := NewDeepSeekClient(config)
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		info := client.GetModelInfo()
		if info.Provider != "deepseek" {
			t.Errorf("Expected provider 'deepseek', got '%s'", info.Provider)
		}
		if info.Name != "deepseek-chat" {
			t.Errorf("Expected model 'deepseek-chat', got '%s'", info.Name)
		}
	})
}

// TestDeepSeekClient_Close tests the Close method functionality
func TestDeepSeekClient_Close(t *testing.T) {
	t.Parallel()

	config := ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	}

	client, err := NewDeepSeekClient(config)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}

	// Verify client is properly initialized
	if client.client == nil {
		t.Fatal("Expected client to be initialized")
	}

	// Test Close method
	err = client.Close()
	if err != nil {
		t.Fatalf("Close method should not return error: %v", err)
	}

	// Verify client is set to nil after Close
	if client.client != nil {
		t.Error("Expected client to be nil after Close")
	}

	// Test that Close can be called multiple times without error
	err = client.Close()
	if err != nil {
		t.Fatalf("Close method should not return error on second call: %v", err)
	}
}

// TestDeepSeekClient_ConvertRequest tests the request conversion functionality
func TestDeepSeekClient_ConvertRequest(t *testing.T) {
	t.Parallel()

	client, err := NewDeepSeekClient(ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	})
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("BasicTextMessage", func(t *testing.T) {
		t.Parallel()
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello, world!"),
			},
		}

		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert request: %v", err)
		}

		if deepseekReq.Model != "deepseek-chat" {
			t.Errorf("Expected model 'deepseek-chat', got '%s'", deepseekReq.Model)
		}
		if len(deepseekReq.Messages) != 1 {
			t.Errorf("Expected 1 message, got %d", len(deepseekReq.Messages))
		}
		if deepseekReq.Messages[0].Role != "user" {
			t.Errorf("Expected role 'user', got '%s'", deepseekReq.Messages[0].Role)
		}
		if deepseekReq.Messages[0].Content != "Hello, world!" {
			t.Errorf("Expected content 'Hello, world!', got '%s'", deepseekReq.Messages[0].Content)
		}
	})

	t.Run("WithTemperature", func(t *testing.T) {
		t.Parallel()
		temp := float32(0.7)
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello"),
			},
			Temperature: &temp,
		}

		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert request: %v", err)
		}

		if deepseekReq.Temperature != 0.7 {
			t.Errorf("Expected temperature 0.7, got %f", deepseekReq.Temperature)
		}
	})

	t.Run("WithMaxTokens", func(t *testing.T) {
		t.Parallel()
		maxTokens := 100
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello"),
			},
			MaxTokens: &maxTokens,
		}

		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert request: %v", err)
		}

		if deepseekReq.MaxTokens != 100 {
			t.Errorf("Expected max tokens 100, got %d", deepseekReq.MaxTokens)
		}
	})

	t.Run("MultipleMessages", func(t *testing.T) {
		t.Parallel()
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleSystem, "You are a helpful assistant."),
				NewTextMessage(RoleUser, "What is 2+2?"),
				NewTextMessage(RoleAssistant, "2+2 equals 4."),
			},
		}

		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert request: %v", err)
		}

		if len(deepseekReq.Messages) != 3 {
			t.Errorf("Expected 3 messages, got %d", len(deepseekReq.Messages))
		}

		expectedRoles := []string{"system", "user", "assistant"}
		expectedContents := []string{"You are a helpful assistant.", "What is 2+2?", "2+2 equals 4."}

		for i, msg := range deepseekReq.Messages {
			if msg.Role != expectedRoles[i] {
				t.Errorf("Message %d: expected role '%s', got '%s'", i, expectedRoles[i], msg.Role)
			}
			if msg.Content != expectedContents[i] {
				t.Errorf("Message %d: expected content '%s', got '%s'", i, expectedContents[i], msg.Content)
			}
		}
	})
}

// TestDeepSeekClient_ConvertResponse tests the response conversion functionality
func TestDeepSeekClient_ConvertResponse(t *testing.T) {
	t.Parallel()

	client, err := NewDeepSeekClient(ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	})
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("BasicResponse", func(t *testing.T) {
		t.Parallel()

		// Import the deepseek package to create test response
		deepseekResp := deepseek.ChatCompletionResponse{
			ID:    "chatcmpl-123",
			Model: "deepseek-chat",
			Choices: []deepseek.Choice{
				{
					Index: 0,
					Message: deepseek.Message{
						Role:    "assistant",
						Content: "Hello! How can I help you today?",
					},
					FinishReason: "stop",
				},
			},
			Usage: deepseek.Usage{
				PromptTokens:     10,
				CompletionTokens: 15,
				TotalTokens:      25,
			},
		}

		response := client.convertResponse(deepseekResp)

		if response.ID != "chatcmpl-123" {
			t.Errorf("Expected ID 'chatcmpl-123', got '%s'", response.ID)
		}
		if response.Model != "deepseek-chat" {
			t.Errorf("Expected model 'deepseek-chat', got '%s'", response.Model)
		}
		if len(response.Choices) != 1 {
			t.Errorf("Expected 1 choice, got %d", len(response.Choices))
		}

		choice := response.Choices[0]
		if choice.Index != 0 {
			t.Errorf("Expected choice index 0, got %d", choice.Index)
		}
		if choice.Message.Role != RoleAssistant {
			t.Errorf("Expected role assistant, got %s", choice.Message.Role)
		}
		if choice.Message.GetText() != "Hello! How can I help you today?" {
			t.Errorf("Expected content 'Hello! How can I help you today?', got '%s'", choice.Message.GetText())
		}
		if choice.FinishReason != "stop" {
			t.Errorf("Expected finish reason 'stop', got '%s'", choice.FinishReason)
		}

		if response.Usage.PromptTokens != 10 {
			t.Errorf("Expected prompt tokens 10, got %d", response.Usage.PromptTokens)
		}
		if response.Usage.CompletionTokens != 15 {
			t.Errorf("Expected completion tokens 15, got %d", response.Usage.CompletionTokens)
		}
		if response.Usage.TotalTokens != 25 {
			t.Errorf("Expected total tokens 25, got %d", response.Usage.TotalTokens)
		}
	})
}

// TestDeepSeekClient_ConvertError tests the error conversion functionality
func TestDeepSeekClient_ConvertError(t *testing.T) {
	t.Parallel()

	client, err := NewDeepSeekClient(ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	})
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("AuthenticationError", func(t *testing.T) {
		t.Parallel()

		originalErr := fmt.Errorf("unauthorized: invalid api key")
		convertedErr := client.convertError(originalErr)

		if convertedErr.Code != "authentication_error" {
			t.Errorf("Expected error code 'authentication_error', got '%s'", convertedErr.Code)
		}
		if convertedErr.Type != "authentication_error" {
			t.Errorf("Expected error type 'authentication_error', got '%s'", convertedErr.Type)
		}
		if convertedErr.StatusCode != 401 {
			t.Errorf("Expected status code 401, got %d", convertedErr.StatusCode)
		}
	})

	t.Run("RateLimitError", func(t *testing.T) {
		t.Parallel()

		originalErr := fmt.Errorf("rate limit exceeded")
		convertedErr := client.convertError(originalErr)

		if convertedErr.Code != "rate_limit_error" {
			t.Errorf("Expected error code 'rate_limit_error', got '%s'", convertedErr.Code)
		}
		if convertedErr.Type != "rate_limit_error" {
			t.Errorf("Expected error type 'rate_limit_error', got '%s'", convertedErr.Type)
		}
		if convertedErr.StatusCode != 429 {
			t.Errorf("Expected status code 429, got %d", convertedErr.StatusCode)
		}
	})

	t.Run("GenericError", func(t *testing.T) {
		t.Parallel()

		originalErr := fmt.Errorf("some generic error")
		convertedErr := client.convertError(originalErr)

		if convertedErr.Code != "api_error" {
			t.Errorf("Expected error code 'api_error', got '%s'", convertedErr.Code)
		}
		if convertedErr.Type != "api_error" {
			t.Errorf("Expected error type 'api_error', got '%s'", convertedErr.Type)
		}
		if convertedErr.Message != "some generic error" {
			t.Errorf("Expected message 'some generic error', got '%s'", convertedErr.Message)
		}
	})
}

// TestDeepSeekClient_ChatCompletion_MockScenario tests ChatCompletion with mock scenarios
func TestDeepSeekClient_ChatCompletion_MockScenario(t *testing.T) {
	t.Parallel()

	t.Run("ValidRequest", func(t *testing.T) {
		t.Parallel()

		client, err := NewDeepSeekClient(ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
		})
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello, world!"),
			},
		}

		// This will fail with a real API call since we're using a test key
		// But we can verify the request conversion works
		_, err = client.ChatCompletion(context.Background(), req)

		// We expect an error since we're using a fake API key
		// But the error should be from the API, not from our conversion logic
		if err == nil {
			t.Error("Expected an error with fake API key")
		}

		// Check that it's not a conversion error
		if llmErr, ok := err.(*Error); ok {
			if llmErr.Code == "security_validation_failed" ||
				llmErr.Code == "tools_not_supported" ||
				llmErr.Code == "multimodal_not_supported" {
				t.Errorf("Unexpected conversion error: %v", err)
			}
		}
	})

	t.Run("SecurityValidationError", func(t *testing.T) {
		t.Parallel()

		client, err := NewDeepSeekClient(ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
		})
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		// Create a message that might fail security validation
		// (This depends on the ValidateMessageSecurity implementation)
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Normal message"),
			},
		}

		// This should work fine with normal content
		_, err = client.ChatCompletion(context.Background(), req)

		// We expect an API error, not a security validation error for normal content
		if err != nil {
			if llmErr, ok := err.(*Error); ok {
				if llmErr.Code == "security_validation_failed" {
					t.Errorf("Unexpected security validation error for normal content: %v", err)
				}
			}
		}
	})
}

// TestDeepSeekClient_StreamingConversion tests the streaming conversion functionality
func TestDeepSeekClient_StreamingConversion(t *testing.T) {
	t.Parallel()

	client, err := NewDeepSeekClient(ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	})
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("ConvertStreamRequest", func(t *testing.T) {
		t.Parallel()
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello, world!"),
			},
			Temperature: func() *float32 { v := float32(0.7); return &v }(),
			MaxTokens:   func() *int { v := 100; return &v }(),
		}

		streamReq, err := client.convertStreamRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert stream request: %v", err)
		}

		if streamReq.Model != "deepseek-chat" {
			t.Errorf("Expected model 'deepseek-chat', got '%s'", streamReq.Model)
		}
		if !streamReq.Stream {
			t.Error("Expected Stream to be true for streaming request")
		}
		if streamReq.Temperature != 0.7 {
			t.Errorf("Expected temperature 0.7, got %f", streamReq.Temperature)
		}
		if streamReq.MaxTokens != 100 {
			t.Errorf("Expected max tokens 100, got %d", streamReq.MaxTokens)
		}
		if len(streamReq.Messages) != 1 {
			t.Errorf("Expected 1 message, got %d", len(streamReq.Messages))
		}
	})

	t.Run("ConvertStreamEvent_Delta", func(t *testing.T) {
		t.Parallel()

		// Create a mock streaming response with delta content
		streamResp := &deepseek.StreamChatCompletionResponse{
			ID:    "chatcmpl-123",
			Model: "deepseek-chat",
			Choices: []deepseek.StreamChoices{
				{
					Index: 0,
					Delta: deepseek.StreamDelta{
						Content: "Hello",
					},
				},
			},
		}

		event := client.convertStreamEvent(streamResp)
		if event == nil {
			t.Fatal("Expected non-nil event")
		}

		if event.Type != "delta" {
			t.Errorf("Expected event type 'delta', got '%s'", event.Type)
		}
		if event.Choice == nil {
			t.Fatal("Expected non-nil choice")
		}
		if event.Choice.Index != 0 {
			t.Errorf("Expected choice index 0, got %d", event.Choice.Index)
		}
		if event.Choice.Delta == nil {
			t.Fatal("Expected non-nil delta")
		}
		if len(event.Choice.Delta.Content) != 1 {
			t.Errorf("Expected 1 content item, got %d", len(event.Choice.Delta.Content))
		}
		if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
			if textContent.GetText() != "Hello" {
				t.Errorf("Expected content 'Hello', got '%s'", textContent.GetText())
			}
		} else {
			t.Error("Expected TextContent type")
		}
	})

	t.Run("ConvertStreamEvent_Done", func(t *testing.T) {
		t.Parallel()

		// Create a mock streaming response with finish reason
		streamResp := &deepseek.StreamChatCompletionResponse{
			ID:    "chatcmpl-123",
			Model: "deepseek-chat",
			Choices: []deepseek.StreamChoices{
				{
					Index:        0,
					FinishReason: "stop",
				},
			},
		}

		event := client.convertStreamEvent(streamResp)
		if event == nil {
			t.Fatal("Expected non-nil event")
		}

		if event.Type != "done" {
			t.Errorf("Expected event type 'done', got '%s'", event.Type)
		}
		if event.Choice == nil {
			t.Fatal("Expected non-nil choice")
		}
		if event.Choice.FinishReason != "stop" {
			t.Errorf("Expected finish reason 'stop', got '%s'", event.Choice.FinishReason)
		}
	})

	t.Run("ConvertStreamEvent_ToolCalls", func(t *testing.T) {
		t.Parallel()

		// Create a mock streaming response with tool calls
		streamResp := &deepseek.StreamChatCompletionResponse{
			ID:    "chatcmpl-123",
			Model: "deepseek-chat",
			Choices: []deepseek.StreamChoices{
				{
					Index: 0,
					Delta: deepseek.StreamDelta{
						ToolCalls: []deepseek.ToolCall{
							{
								Index: 0,
								ID:    "call_123",
								Type:  "function",
								Function: deepseek.ToolCallFunction{
									Name:      "get_weather",
									Arguments: `{"location": "New York"}`,
								},
							},
						},
					},
				},
			},
		}

		event := client.convertStreamEvent(streamResp)
		if event == nil {
			t.Fatal("Expected non-nil event")
		}

		if event.Type != "delta" {
			t.Errorf("Expected event type 'delta', got '%s'", event.Type)
		}
		if event.Choice.Delta == nil {
			t.Fatal("Expected non-nil delta")
		}
		if len(event.Choice.Delta.ToolCalls) != 1 {
			t.Errorf("Expected 1 tool call, got %d", len(event.Choice.Delta.ToolCalls))
		}

		toolCall := event.Choice.Delta.ToolCalls[0]
		if toolCall.ID != "call_123" {
			t.Errorf("Expected tool call ID 'call_123', got '%s'", toolCall.ID)
		}
		if toolCall.Type != "function" {
			t.Errorf("Expected tool call type 'function', got '%s'", toolCall.Type)
		}
		if toolCall.Function == nil {
			t.Fatal("Expected non-nil function")
		}
		if toolCall.Function.Name != "get_weather" {
			t.Errorf("Expected function name 'get_weather', got '%s'", toolCall.Function.Name)
		}
		if toolCall.Function.Arguments != `{"location": "New York"}` {
			t.Errorf("Expected arguments '{\"location\": \"New York\"}', got '%s'", toolCall.Function.Arguments)
		}
	})

	t.Run("ConvertStreamEvent_Empty", func(t *testing.T) {
		t.Parallel()

		// Test with nil response
		event := client.convertStreamEvent(nil)
		if event != nil {
			t.Error("Expected nil event for nil response")
		}

		// Test with empty choices
		streamResp := &deepseek.StreamChatCompletionResponse{
			ID:      "chatcmpl-123",
			Model:   "deepseek-chat",
			Choices: []deepseek.StreamChoices{},
		}

		event = client.convertStreamEvent(streamResp)
		if event != nil {
			t.Error("Expected nil event for empty choices")
		}
	})
}

// TestDeepSeekClient_StreamChatCompletion_MockScenario tests StreamChatCompletion with mock scenarios
func TestDeepSeekClient_StreamChatCompletion_MockScenario(t *testing.T) {
	t.Parallel()

	t.Run("ValidRequest", func(t *testing.T) {
		t.Parallel()

		client, err := NewDeepSeekClient(ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-chat",
		})
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello, world!"),
			},
		}

		// This will fail with a real API call since we're using a test key
		// But we can verify the request conversion and channel creation works
		ch, err := client.StreamChatCompletion(context.Background(), req)

		// We expect an error since we're using a fake API key
		// But the error should be from the API, not from our conversion logic
		if err == nil {
			// If no error, we should get a channel
			if ch == nil {
				t.Error("Expected non-nil channel when no error")
			} else {
				// Try to read from the channel (should get an error event)
				select {
				case event := <-ch:
					if event.IsError() {
						// This is expected - API error due to fake key
						t.Logf("Got expected API error: %v", event.Error)
					} else {
						t.Logf("Got unexpected event type: %s", event.Type)
					}
				case <-time.After(1 * time.Second):
					t.Log("Timeout waiting for stream event (expected with fake API key)")
				}
			}
		} else {
			// Check that it's not a conversion error
			if llmErr, ok := err.(*Error); ok {
				if llmErr.Code == "security_validation_failed" ||
					llmErr.Code == "tools_not_supported" ||
					llmErr.Code == "multimodal_not_supported" {
					t.Errorf("Unexpected conversion error: %v", err)
				}
			}
		}
	})
}

// TestDeepSeekClient_ToolSupport tests tool functionality
func TestDeepSeekClient_ToolSupport(t *testing.T) {
	t.Parallel()

	config := ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	}

	client, err := NewDeepSeekClient(config)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("ConvertToolParameters_ValidParameters", func(t *testing.T) {
		t.Parallel()

		// Test with typical JSON schema parameters
		params := map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"location": map[string]interface{}{
					"type":        "string",
					"description": "The city and state",
				},
				"unit": map[string]interface{}{
					"type": "string",
					"enum": []string{"celsius", "fahrenheit"},
				},
			},
			"required": []string{"location"},
		}

		result := client.convertToolParameters(params)
		if result == nil {
			t.Fatal("Expected non-nil result")
		}

		if result.Type != "object" {
			t.Errorf("Expected type 'object', got '%s'", result.Type)
		}

		if result.Properties == nil {
			t.Fatal("Expected non-nil properties")
		}

		if len(result.Required) != 1 || result.Required[0] != "location" {
			t.Errorf("Expected required=['location'], got %v", result.Required)
		}

		// Check properties structure
		props, ok := result.Properties["location"].(map[string]interface{})
		if !ok {
			t.Fatal("Expected location property to be a map")
		}
		if props["type"] != "string" {
			t.Errorf("Expected location type 'string', got %v", props["type"])
		}
	})

	t.Run("ConvertToolParameters_NilParameters", func(t *testing.T) {
		t.Parallel()

		result := client.convertToolParameters(nil)
		if result != nil {
			t.Error("Expected nil result for nil parameters")
		}
	})

	t.Run("ConvertToolParameters_InvalidType", func(t *testing.T) {
		t.Parallel()

		// Test with non-map parameters
		result := client.convertToolParameters("invalid")
		if result == nil {
			t.Fatal("Expected non-nil result")
		}
		if result.Type != "object" {
			t.Errorf("Expected default type 'object', got '%s'", result.Type)
		}
	})

	t.Run("ConvertToolParameters_RequiredAsInterfaceSlice", func(t *testing.T) {
		t.Parallel()

		params := map[string]interface{}{
			"type":     "object",
			"required": []interface{}{"param1", "param2"},
		}

		result := client.convertToolParameters(params)
		if result == nil {
			t.Fatal("Expected non-nil result")
		}

		if len(result.Required) != 2 {
			t.Errorf("Expected 2 required fields, got %d", len(result.Required))
		}
		if result.Required[0] != "param1" || result.Required[1] != "param2" {
			t.Errorf("Expected required=['param1', 'param2'], got %v", result.Required)
		}
	})

	t.Run("ConvertRequest_WithTools", func(t *testing.T) {
		t.Parallel()

		// Create a request with tools
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "What's the weather?"),
			},
			Tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather",
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"location": map[string]interface{}{
									"type": "string",
								},
							},
							"required": []string{"location"},
						},
					},
				},
			},
		}

		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert request: %v", err)
		}

		if len(deepseekReq.Tools) != 1 {
			t.Errorf("Expected 1 tool, got %d", len(deepseekReq.Tools))
		}

		tool := deepseekReq.Tools[0]
		if tool.Type != "function" {
			t.Errorf("Expected tool type 'function', got '%s'", tool.Type)
		}
		if tool.Function.Name != "get_weather" {
			t.Errorf("Expected function name 'get_weather', got '%s'", tool.Function.Name)
		}
		if tool.Function.Description != "Get current weather" {
			t.Errorf("Expected description 'Get current weather', got '%s'", tool.Function.Description)
		}

		if tool.Function.Parameters == nil {
			t.Fatal("Expected non-nil parameters")
		}
		if tool.Function.Parameters.Type != "object" {
			t.Errorf("Expected parameters type 'object', got '%s'", tool.Function.Parameters.Type)
		}
	})

	t.Run("ConvertRequest_ToolsNotSupported", func(t *testing.T) {
		t.Parallel()

		// Create client with a model that doesn't support tools
		config := ClientConfig{
			Provider: "deepseek",
			APIKey:   "test-key",
			Model:    "deepseek-unsupported", // This should not support tools
		}

		client, err := NewDeepSeekClient(config)
		if err != nil {
			t.Fatalf("Failed to create DeepSeek client: %v", err)
		}
		defer func() { _ = client.Close() }()

		req := ChatRequest{
			Model: "deepseek-unsupported",
			Messages: []Message{
				NewTextMessage(RoleUser, "Test"),
			},
			Tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name: "test_function",
					},
				},
			},
		}

		_, err = client.convertRequest(req)
		if err == nil {
			t.Error("Expected error for unsupported tools")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "tools_not_supported" {
				t.Errorf("Expected error code 'tools_not_supported', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ConvertResponse_WithToolCalls", func(t *testing.T) {
		t.Parallel()

		// Create a mock DeepSeek response with tool calls
		deepseekResp := deepseek.ChatCompletionResponse{
			ID:    "chatcmpl-123",
			Model: "deepseek-chat",
			Choices: []deepseek.Choice{
				{
					Index: 0,
					Message: deepseek.Message{
						Role:    "assistant",
						Content: "",
						ToolCalls: []deepseek.ToolCall{
							{
								ID:   "call_123",
								Type: "function",
								Function: deepseek.ToolCallFunction{
									Name:      "get_weather",
									Arguments: `{"location": "New York"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: deepseek.Usage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}

		response := client.convertResponse(deepseekResp)
		if response == nil {
			t.Fatal("Expected non-nil response")
		}

		if len(response.Choices) != 1 {
			t.Errorf("Expected 1 choice, got %d", len(response.Choices))
		}

		choice := response.Choices[0]
		if len(choice.Message.ToolCalls) != 1 {
			t.Errorf("Expected 1 tool call, got %d", len(choice.Message.ToolCalls))
		}

		toolCall := choice.Message.ToolCalls[0]
		if toolCall.ID != "call_123" {
			t.Errorf("Expected tool call ID 'call_123', got '%s'", toolCall.ID)
		}
		if toolCall.Type != "function" {
			t.Errorf("Expected tool call type 'function', got '%s'", toolCall.Type)
		}
		if toolCall.Function.Name != "get_weather" {
			t.Errorf("Expected function name 'get_weather', got '%s'", toolCall.Function.Name)
		}
		if toolCall.Function.Arguments != `{"location": "New York"}` {
			t.Errorf("Expected arguments '{\"location\": \"New York\"}', got '%s'", toolCall.Function.Arguments)
		}
	})

	t.Run("ConvertStreamRequest_WithTools", func(t *testing.T) {
		t.Parallel()

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				NewTextMessage(RoleUser, "What's the weather?"),
			},
			Tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "get_weather",
						Description: "Get current weather",
						Parameters: map[string]interface{}{
							"type": "object",
							"properties": map[string]interface{}{
								"location": map[string]interface{}{
									"type": "string",
								},
							},
						},
					},
				},
			},
		}

		deepseekReq, err := client.convertStreamRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert stream request: %v", err)
		}

		if len(deepseekReq.Tools) != 1 {
			t.Errorf("Expected 1 tool, got %d", len(deepseekReq.Tools))
		}

		if !deepseekReq.Stream {
			t.Error("Expected Stream to be true")
		}

		tool := deepseekReq.Tools[0]
		if tool.Function.Name != "get_weather" {
			t.Errorf("Expected function name 'get_weather', got '%s'", tool.Function.Name)
		}
	})
}

// TestDeepSeekClient_ModelCapabilities tests model capability detection
func TestDeepSeekClient_ModelCapabilities(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name           string
		model          string
		expectedTools  bool
		expectedVision bool
		expectedFiles  bool
		expectedTokens int
	}{
		{
			name:           "deepseek-chat supports tools",
			model:          "deepseek-chat",
			expectedTools:  true,
			expectedVision: false,
			expectedFiles:  false,
			expectedTokens: 32768,
		},
		{
			name:           "deepseek-coder supports tools",
			model:          "deepseek-coder",
			expectedTools:  true,
			expectedVision: false,
			expectedFiles:  false,
			expectedTokens: 32768,
		},
		{
			name:           "deepseek-unknown conservative defaults",
			model:          "deepseek-unknown-model",
			expectedTools:  false,
			expectedVision: false,
			expectedFiles:  false,
			expectedTokens: 32768,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			config := ClientConfig{
				Provider: "deepseek",
				APIKey:   "test-key",
				Model:    tc.model,
			}

			client, err := NewDeepSeekClient(config)
			if err != nil {
				t.Fatalf("Failed to create DeepSeek client: %v", err)
			}
			defer func() { _ = client.Close() }()

			modelInfo := client.GetModelInfo()

			if modelInfo.SupportsTools != tc.expectedTools {
				t.Errorf("Expected SupportsTools=%v, got %v", tc.expectedTools, modelInfo.SupportsTools)
			}
			if modelInfo.SupportsVision != tc.expectedVision {
				t.Errorf("Expected SupportsVision=%v, got %v", tc.expectedVision, modelInfo.SupportsVision)
			}
			if modelInfo.SupportsFiles != tc.expectedFiles {
				t.Errorf("Expected SupportsFiles=%v, got %v", tc.expectedFiles, modelInfo.SupportsFiles)
			}
			if modelInfo.MaxTokens != tc.expectedTokens {
				t.Errorf("Expected MaxTokens=%d, got %d", tc.expectedTokens, modelInfo.MaxTokens)
			}
			if modelInfo.Provider != "deepseek" {
				t.Errorf("Expected Provider='deepseek', got '%s'", modelInfo.Provider)
			}
			if modelInfo.Name != tc.model {
				t.Errorf("Expected Name='%s', got '%s'", tc.model, modelInfo.Name)
			}
		})
	}
}

// TestDeepSeekClient_MultiModalContent tests multi-modal content support
func TestDeepSeekClient_MultiModalContent(t *testing.T) {
	t.Parallel()

	config := ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	}

	client, err := NewDeepSeekClient(config)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("ValidateImageContent_VisionNotSupported", func(t *testing.T) {
		t.Parallel()

		// Create image content
		imageData := []byte("fake-image-data")
		img := NewImageContentFromBytes(imageData, "image/jpeg")

		// Get model info (should not support vision by default)
		modelInfo := client.GetModelInfo()

		err := client.validateImageContent(img, modelInfo)
		if err == nil {
			t.Error("Expected error for vision not supported")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "vision_not_supported" {
				t.Errorf("Expected error code 'vision_not_supported', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidateFileContent_FilesNotSupported", func(t *testing.T) {
		t.Parallel()

		// Create file content
		fileData := []byte("test file content")
		file := NewFileContentFromBytes(fileData, "test.txt", "text/plain")

		// Get model info (should not support files by default)
		modelInfo := client.GetModelInfo()

		err := client.validateFileContent(file, modelInfo)
		if err == nil {
			t.Error("Expected error for files not supported")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "files_not_supported" {
				t.Errorf("Expected error code 'files_not_supported', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidateImageContent_SizeExceeded", func(t *testing.T) {
		t.Parallel()

		// Create oversized image content
		config := DefaultSecurityConfig()
		oversizedData := make([]byte, config.MaxImageSize+1)
		img := NewImageContentFromBytes(oversizedData, "image/jpeg")

		// Mock model info to support vision
		modelInfo := ModelInfo{
			SupportsVision: true,
		}

		err := client.validateImageContent(img, modelInfo)
		if err == nil {
			t.Error("Expected error for image size exceeded")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "image_size_exceeded" {
				t.Errorf("Expected error code 'image_size_exceeded', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidateFileContent_SizeExceeded", func(t *testing.T) {
		t.Parallel()

		// Create oversized file content
		config := DefaultSecurityConfig()
		oversizedData := make([]byte, config.MaxFileSize+1)
		file := NewFileContentFromBytes(oversizedData, "large.txt", "text/plain")

		// Mock model info to support files
		modelInfo := ModelInfo{
			SupportsFiles: true,
		}

		err := client.validateFileContent(file, modelInfo)
		if err == nil {
			t.Error("Expected error for file size exceeded")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "file_size_exceeded" {
				t.Errorf("Expected error code 'file_size_exceeded', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidateImageContent_InvalidMimeType", func(t *testing.T) {
		t.Parallel()

		// Create image with invalid MIME type
		imageData := []byte("fake-image-data")
		img := NewImageContentFromBytes(imageData, "application/octet-stream")

		// Mock model info to support vision
		modelInfo := ModelInfo{
			SupportsVision: true,
		}

		err := client.validateImageContent(img, modelInfo)
		if err == nil {
			t.Error("Expected error for unsupported image type")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "unsupported_image_type" {
				t.Errorf("Expected error code 'unsupported_image_type', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidateFileContent_InvalidMimeType", func(t *testing.T) {
		t.Parallel()

		// Create file with invalid MIME type
		fileData := []byte("test content")
		file := NewFileContentFromBytes(fileData, "test.exe", "application/x-executable")

		// Mock model info to support files
		modelInfo := ModelInfo{
			SupportsFiles: true,
		}

		err := client.validateFileContent(file, modelInfo)
		if err == nil {
			t.Error("Expected error for unsupported file type")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "unsupported_file_type" {
				t.Errorf("Expected error code 'unsupported_file_type', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ConvertImageContent_WithURL", func(t *testing.T) {
		t.Parallel()

		img := NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg")
		modelInfo := ModelInfo{SupportsVision: true}

		result, err := client.convertImageContent(img, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert image content: %v", err)
		}

		expected := "[Image: https://example.com/image.jpg, Type: image/jpeg]"
		if result != expected {
			t.Errorf("Expected '%s', got '%s'", expected, result)
		}
	})

	t.Run("ConvertImageContent_WithData", func(t *testing.T) {
		t.Parallel()

		imageData := []byte("fake-image-data")
		img := NewImageContentFromBytes(imageData, "image/png")
		img.SetDimensions(800, 600)

		modelInfo := ModelInfo{SupportsVision: true}

		result, err := client.convertImageContent(img, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert image content: %v", err)
		}

		// Should contain the MIME type, size, and dimensions
		if !contains(result, "image/png") {
			t.Errorf("Expected result to contain 'image/png', got '%s'", result)
		}
		if !contains(result, "800x600") {
			t.Errorf("Expected result to contain dimensions '800x600', got '%s'", result)
		}
		if !contains(result, "base64 data") {
			t.Errorf("Expected result to contain 'base64 data', got '%s'", result)
		}
	})

	t.Run("ConvertFileContent_TextFile", func(t *testing.T) {
		t.Parallel()

		fileContent := "Hello, world!\nThis is a test file."
		file := NewFileContentFromBytes([]byte(fileContent), "test.txt", "text/plain")

		modelInfo := ModelInfo{SupportsFiles: true}

		result, err := client.convertFileContent(file, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert file content: %v", err)
		}

		// Should contain the filename, MIME type, and actual content
		if !contains(result, "test.txt") {
			t.Errorf("Expected result to contain 'test.txt', got '%s'", result)
		}
		if !contains(result, "text/plain") {
			t.Errorf("Expected result to contain 'text/plain', got '%s'", result)
		}
		if !contains(result, fileContent) {
			t.Errorf("Expected result to contain file content, got '%s'", result)
		}
	})

	t.Run("ConvertFileContent_JSONFile", func(t *testing.T) {
		t.Parallel()

		jsonContent := `{"name": "test", "value": 123}`
		file := NewFileContentFromBytes([]byte(jsonContent), "data.json", "application/json")

		modelInfo := ModelInfo{SupportsFiles: true}

		result, err := client.convertFileContent(file, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert file content: %v", err)
		}

		// Should contain the filename, MIME type, and actual JSON content
		if !contains(result, "data.json") {
			t.Errorf("Expected result to contain 'data.json', got '%s'", result)
		}
		if !contains(result, "application/json") {
			t.Errorf("Expected result to contain 'application/json', got '%s'", result)
		}
		if !contains(result, jsonContent) {
			t.Errorf("Expected result to contain JSON content, got '%s'", result)
		}
	})

	t.Run("ConvertFileContent_PDFFile", func(t *testing.T) {
		t.Parallel()

		pdfData := []byte("%PDF-1.4 fake pdf content")
		file := NewFileContentFromBytes(pdfData, "document.pdf", "application/pdf")

		modelInfo := ModelInfo{SupportsFiles: true}

		result, err := client.convertFileContent(file, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert file content: %v", err)
		}

		// Should contain metadata but not the raw PDF content
		if !contains(result, "document.pdf") {
			t.Errorf("Expected result to contain 'document.pdf', got '%s'", result)
		}
		if !contains(result, "PDF File") {
			t.Errorf("Expected result to contain 'PDF File', got '%s'", result)
		}
		// Should not contain the raw PDF data
		if contains(result, "%PDF-1.4") {
			t.Errorf("Expected result to not contain raw PDF data, got '%s'", result)
		}
	})

	t.Run("ConvertFileContent_WithURL", func(t *testing.T) {
		t.Parallel()

		file := NewFileContentFromURL("https://example.com/file.txt", "file.txt", "text/plain", 1024)

		modelInfo := ModelInfo{SupportsFiles: true}

		result, err := client.convertFileContent(file, modelInfo)
		if err != nil {
			t.Fatalf("Failed to convert file content: %v", err)
		}

		// Should contain the URL and metadata
		if !contains(result, "https://example.com/file.txt") {
			t.Errorf("Expected result to contain URL, got '%s'", result)
		}
		if !contains(result, "file.txt") {
			t.Errorf("Expected result to contain filename, got '%s'", result)
		}
		if !contains(result, "text/plain") {
			t.Errorf("Expected result to contain MIME type, got '%s'", result)
		}
	})

	t.Run("ValidateMessageSize", func(t *testing.T) {
		t.Parallel()

		// Test with normal size
		err := client.validateMessageSize(1024)
		if err != nil {
			t.Errorf("Expected no error for normal size, got: %v", err)
		}

		// Test with oversized message
		config := DefaultSecurityConfig()
		err = client.validateMessageSize(config.MaxTotalSize + 1)
		if err == nil {
			t.Error("Expected error for oversized message")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "message_size_exceeded" {
				t.Errorf("Expected error code 'message_size_exceeded', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ConvertMessage_WithMultiModalContent", func(t *testing.T) {
		t.Parallel()

		// Create a message with mixed content types
		msg := Message{
			Role: RoleUser,
			Content: []MessageContent{
				NewTextContent("Please analyze this image and file:"),
				// Note: These will fail validation since the model doesn't support them
				// but we can test the error handling
			},
		}

		// Test with text-only content (should work)
		deepseekMsg, err := client.convertMessage(msg)
		if err != nil {
			t.Fatalf("Failed to convert message with text content: %v", err)
		}

		if deepseekMsg.Content != "Please analyze this image and file:" {
			t.Errorf("Expected text content, got '%s'", deepseekMsg.Content)
		}
	})
}

// TestDeepSeekClient_MultiModalIntegration tests end-to-end multi-modal content handling
func TestDeepSeekClient_MultiModalIntegration(t *testing.T) {
	t.Parallel()

	config := ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	}

	client, err := NewDeepSeekClient(config)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("ConvertRequest_WithUnsupportedMultiModal", func(t *testing.T) {
		t.Parallel()

		// Create a request with multi-modal content that should fail validation
		imageData := []byte("fake-image-data")
		img := NewImageContentFromBytes(imageData, "image/jpeg")

		fileData := []byte("test file content")
		file := NewFileContentFromBytes(fileData, "test.txt", "text/plain")

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				{
					Role: RoleUser,
					Content: []MessageContent{
						NewTextContent("Please analyze this image and file:"),
						img,  // This should fail - vision not supported
						file, // This should fail - files not supported
					},
				},
			},
		}

		// The convertRequest should fail because the model doesn't support vision/files
		_, err := client.convertRequest(req)
		if err == nil {
			t.Error("Expected error for unsupported multi-modal content")
		}

		// Should be a security validation error (security validation happens first)
		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "security_validation_failed" {
				t.Errorf("Expected error code 'security_validation_failed', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ConvertRequest_TextOnlySuccess", func(t *testing.T) {
		t.Parallel()

		// Create a request with only text content (should work)
		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				{
					Role: RoleUser,
					Content: []MessageContent{
						NewTextContent("Hello, how are you?"),
						NewTextContent("This is a follow-up question."),
					},
				},
			},
		}

		// This should succeed
		deepseekReq, err := client.convertRequest(req)
		if err != nil {
			t.Fatalf("Failed to convert text-only request: %v", err)
		}

		if len(deepseekReq.Messages) != 1 {
			t.Errorf("Expected 1 message, got %d", len(deepseekReq.Messages))
		}

		// Content should be combined
		expectedContent := "Hello, how are you?\n\nThis is a follow-up question."
		if deepseekReq.Messages[0].Content != expectedContent {
			t.Errorf("Expected combined content '%s', got '%s'", expectedContent, deepseekReq.Messages[0].Content)
		}
	})

	t.Run("MessageSizeValidation", func(t *testing.T) {
		t.Parallel()

		// Create a message that exceeds size limits
		config := DefaultSecurityConfig()
		largeText := make([]byte, config.MaxTotalSize+1)
		for i := range largeText {
			largeText[i] = 'A'
		}

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				{
					Role: RoleUser,
					Content: []MessageContent{
						NewTextContent(string(largeText)),
					},
				},
			},
		}

		// This should fail due to size limits
		_, err := client.convertRequest(req)
		if err == nil {
			t.Error("Expected error for oversized message")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			// Could be either security validation or message size validation
			if llmErr.Code != "message_size_exceeded" && llmErr.Code != "security_validation_failed" {
				t.Errorf("Expected error code 'message_size_exceeded' or 'security_validation_failed', got '%s'", llmErr.Code)
			}
		}
	})
}

// TestDeepSeekClient_ValidationOrder tests that validation happens in the correct order
func TestDeepSeekClient_ValidationOrder(t *testing.T) {
	t.Parallel()

	config := ClientConfig{
		Provider: "deepseek",
		APIKey:   "test-key",
		Model:    "deepseek-chat",
	}

	client, err := NewDeepSeekClient(config)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek client: %v", err)
	}
	defer func() { _ = client.Close() }()

	t.Run("ValidImageContent_VisionNotSupported", func(t *testing.T) {
		t.Parallel()

		// Create valid image content that passes security validation
		// but should fail model capability validation
		imageData := []byte{
			0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, // JPEG header
			0x00, 0x01, 0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00,
		}
		img := NewImageContentFromBytes(imageData, "image/jpeg")

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				{
					Role: RoleUser,
					Content: []MessageContent{
						NewTextContent("Please analyze this image:"),
						img,
					},
				},
			},
		}

		// This should fail with vision not supported (after passing security validation)
		_, err := client.convertRequest(req)
		if err == nil {
			t.Error("Expected error for vision not supported")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "vision_not_supported" {
				t.Errorf("Expected error code 'vision_not_supported', got '%s'", llmErr.Code)
			}
		}
	})

	t.Run("ValidFileContent_FilesNotSupported", func(t *testing.T) {
		t.Parallel()

		// Create valid file content that passes security validation
		// but should fail model capability validation
		fileContent := "This is a valid text file content."
		file := NewFileContentFromBytes([]byte(fileContent), "test.txt", "text/plain")

		req := ChatRequest{
			Model: "deepseek-chat",
			Messages: []Message{
				{
					Role: RoleUser,
					Content: []MessageContent{
						NewTextContent("Please analyze this file:"),
						file,
					},
				},
			},
		}

		// This should fail with files not supported (after passing security validation)
		_, err := client.convertRequest(req)
		if err == nil {
			t.Error("Expected error for files not supported")
		}

		llmErr, ok := err.(*Error)
		if !ok {
			t.Errorf("Expected *Error, got %T", err)
		} else {
			if llmErr.Code != "files_not_supported" {
				t.Errorf("Expected error code 'files_not_supported', got '%s'", llmErr.Code)
			}
		}
	})
}
