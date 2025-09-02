package llm

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/revrost/go-openrouter"
)

// TestNewOpenRouterClient tests client creation with various configurations
func TestNewOpenRouterClient(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		config      ClientConfig
		expectError bool
		errorCode   string
	}{
		{
			name: "valid_basic_config",
			config: ClientConfig{
				Provider: "openrouter",
				APIKey:   "sk-or-test-key",
				Model:    "openai/gpt-3.5-turbo",
			},
			expectError: false,
		},
		{
			name: "valid_config_with_base_url",
			config: ClientConfig{
				Provider: "openrouter",
				APIKey:   "sk-or-test-key",
				Model:    "openai/gpt-4",
				BaseURL:  "https://custom.openrouter.ai",
			},
			expectError: false,
		},
		{
			name: "valid_config_with_extra_params",
			config: ClientConfig{
				Provider: "openrouter",
				APIKey:   "sk-or-test-key",
				Model:    "anthropic/claude-3-sonnet",
				Extra: map[string]string{
					"site_url": "https://myapp.com",
					"app_name": "MyTestApp",
				},
			},
			expectError: false,
		},
		{
			name: "missing_api_key",
			config: ClientConfig{
				Provider: "openrouter",
				APIKey:   "",
				Model:    "openai/gpt-3.5-turbo",
			},
			expectError: true,
			errorCode:   "missing_api_key",
		},
		{
			name: "empty_model_allowed",
			config: ClientConfig{
				Provider: "openrouter",
				APIKey:   "sk-or-test-key",
				Model:    "",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewOpenRouterClient(tt.config)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
					return
				}

				if llmErr, ok := err.(*Error); ok {
					if llmErr.Code != tt.errorCode {
						t.Errorf("Expected error code %s, got %s", tt.errorCode, llmErr.Code)
					}
					if llmErr.Type != "authentication_error" {
						t.Errorf("Expected error type 'authentication_error', got %s", llmErr.Type)
					}
				} else {
					t.Errorf("Expected *Error type, got %T", err)
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if client == nil {
				t.Error("Client should not be nil")
				return
			}

			// Verify client properties
			if client.provider != "openrouter" {
				t.Errorf("Expected provider 'openrouter', got %s", client.provider)
			}

			if client.model != tt.config.Model {
				t.Errorf("Expected model %s, got %s", tt.config.Model, client.model)
			}

			if !reflect.DeepEqual(client.config, tt.config) {
				t.Errorf("Config not stored correctly")
			}

			// Test Close method
			if err := client.Close(); err != nil {
				t.Errorf("Close() should not return error: %v", err)
			}

			// Verify client is cleaned up
			if client.client != nil {
				t.Error("Client should be nil after Close()")
			}
		})
	}
}

// TestOpenRouterClient_Close tests the Close method
func TestOpenRouterClient_Close(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}

	// Verify client is initially set
	if client.client == nil {
		t.Error("Client should not be nil initially")
	}

	// Test first Close call
	err = client.Close()
	if err != nil {
		t.Errorf("Close() should not return error: %v", err)
	}

	// Verify client is cleaned up
	if client.client != nil {
		t.Error("Client should be nil after Close()")
	}

	// Test second Close call (should be safe to call multiple times)
	err = client.Close()
	if err != nil {
		t.Errorf("Second Close() should not return error: %v", err)
	}

	// Verify client is still nil
	if client.client != nil {
		t.Error("Client should remain nil after second Close()")
	}
}

// TestOpenRouterClient_GetModelInfo tests model info retrieval
func TestOpenRouterClient_GetModelInfo(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		model    string
		expected ModelInfo
	}{
		{
			name:  "gpt-4o_model",
			model: "openai/gpt-4o",
			expected: ModelInfo{
				Name:              "openai/gpt-4o",
				Provider:          "openrouter",
				MaxTokens:         128000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			name:  "claude-3_model",
			model: "anthropic/claude-3-sonnet",
			expected: ModelInfo{
				Name:              "anthropic/claude-3-sonnet",
				Provider:          "openrouter",
				MaxTokens:         200000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			name:  "unknown_model",
			model: "unknown/model",
			expected: ModelInfo{
				Name:              "unknown/model",
				Provider:          "openrouter",
				MaxTokens:         4096,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewOpenRouterClient(ClientConfig{
				Provider: "openrouter",
				APIKey:   "test-key",
				Model:    tt.model,
			})
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer func() { _ = client.Close() }()

			info := client.GetModelInfo()

			if !reflect.DeepEqual(info, tt.expected) {
				t.Errorf("Expected model info %+v, got %+v", tt.expected, info)
			}
		})
	}
}

// TestOpenRouterClient_convertRequest tests request conversion
func TestOpenRouterClient_convertRequest(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name        string
		request     ChatRequest
		expectError bool
		validate    func(t *testing.T, req openrouter.ChatCompletionRequest)
	}{
		{
			name: "basic_text_request",
			request: ChatRequest{
				Model: "openai/gpt-4",
				Messages: []Message{
					NewTextMessage(RoleUser, "Hello, world!"),
				},
			},
			expectError: false,
			validate: func(t *testing.T, req openrouter.ChatCompletionRequest) {
				if req.Model != "openai/gpt-4" {
					t.Errorf("Expected model 'openai/gpt-4', got %s", req.Model)
				}
				if len(req.Messages) != 1 {
					t.Errorf("Expected 1 message, got %d", len(req.Messages))
				}
				if req.Messages[0].Content.Text != "Hello, world!" {
					t.Errorf("Expected message 'Hello, world!', got %s", req.Messages[0].Content.Text)
				}
			},
		},
		{
			name: "request_with_parameters",
			request: ChatRequest{
				Model:       "openai/gpt-3.5-turbo",
				Temperature: floatPtr(0.7),
				MaxTokens:   intPtr(1000),
				TopP:        floatPtr(0.9),
				Messages: []Message{
					NewTextMessage(RoleUser, "Test message"),
				},
			},
			expectError: false,
			validate: func(t *testing.T, req openrouter.ChatCompletionRequest) {
				if req.Temperature != 0.7 {
					t.Errorf("Expected temperature 0.7, got %f", req.Temperature)
				}
				if req.MaxTokens != 1000 {
					t.Errorf("Expected max tokens 1000, got %d", req.MaxTokens)
				}
				if req.TopP != 0.9 {
					t.Errorf("Expected top_p 0.9, got %f", req.TopP)
				}
			},
		},
		{
			name: "request_with_tools",
			request: ChatRequest{
				Model: "openai/gpt-4",
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
										"type":        "string",
										"description": "City name",
									},
								},
								"required": []string{"location"},
							},
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, req openrouter.ChatCompletionRequest) {
				if len(req.Tools) != 1 {
					t.Errorf("Expected 1 tool, got %d", len(req.Tools))
				}
				if req.Tools[0].Function.Name != "get_weather" {
					t.Errorf("Expected tool name 'get_weather', got %s", req.Tools[0].Function.Name)
				}
			},
		},
		{
			name: "request_with_unsupported_tools",
			request: ChatRequest{
				Model: "meta-llama/llama-3-8b", // Model that doesn't support tools
				Messages: []Message{
					NewTextMessage(RoleUser, "Test"),
				},
				Tools: []Tool{
					{
						Type: "function",
						Function: ToolFunction{
							Name:        "test_function",
							Description: "Test function",
							Parameters: map[string]interface{}{
								"type": "object",
							},
						},
					},
				},
			},
			expectError: true,
		},
		{
			name: "empty_model_uses_client_model",
			request: ChatRequest{
				Messages: []Message{
					NewTextMessage(RoleUser, "Test"),
				},
			},
			expectError: false,
			validate: func(t *testing.T, req openrouter.ChatCompletionRequest) {
				if req.Model != "openai/gpt-3.5-turbo" {
					t.Errorf("Expected model 'openai/gpt-3.5-turbo', got %s", req.Model)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Create client with appropriate model for the test
			testClient := client
			if tt.name == "request_with_unsupported_tools" {
				testClient, _ = NewOpenRouterClient(ClientConfig{
					Provider: "openrouter",
					APIKey:   "test-key",
					Model:    "meta-llama/llama-3-8b",
				})
				defer func() { _ = testClient.Close() }()
			}

			req, err := testClient.convertRequest(tt.request)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.validate != nil {
				tt.validate(t, req)
			}
		})
	}
}

// TestOpenRouterClient_convertMessage tests message conversion
func TestOpenRouterClient_convertMessage(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-4o", // Vision-capable model
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name        string
		message     Message
		expectError bool
		validate    func(t *testing.T, msg openrouter.ChatCompletionMessage)
	}{
		{
			name:        "simple_text_message",
			message:     NewTextMessage(RoleUser, "Hello"),
			expectError: false,
			validate: func(t *testing.T, msg openrouter.ChatCompletionMessage) {
				if msg.Role != "user" {
					t.Errorf("Expected role 'user', got %s", msg.Role)
				}
				if msg.Content.Text != "Hello" {
					t.Errorf("Expected content 'Hello', got %s", msg.Content.Text)
				}
			},
		},
		{
			name: "message_with_tool_calls",
			message: Message{
				Role: RoleAssistant,
				Content: []MessageContent{
					NewTextContent("I'll check the weather for you."),
				},
				ToolCalls: []ToolCall{
					{
						ID:   "call_123",
						Type: "function",
						Function: ToolCallFunction{
							Name:      "get_weather",
							Arguments: `{"location": "New York"}`,
						},
					},
				},
			},
			expectError: false,
			validate: func(t *testing.T, msg openrouter.ChatCompletionMessage) {
				if len(msg.ToolCalls) != 1 {
					t.Errorf("Expected 1 tool call, got %d", len(msg.ToolCalls))
				}
				if msg.ToolCalls[0].ID != "call_123" {
					t.Errorf("Expected tool call ID 'call_123', got %s", msg.ToolCalls[0].ID)
				}
			},
		},
		{
			name: "tool_response_message",
			message: Message{
				Role:       RoleTool,
				ToolCallID: "call_123",
				Content: []MessageContent{
					NewTextContent("Weather is sunny, 75°F"),
				},
			},
			expectError: false,
			validate: func(t *testing.T, msg openrouter.ChatCompletionMessage) {
				if msg.ToolCallID != "call_123" {
					t.Errorf("Expected tool call ID 'call_123', got %s", msg.ToolCallID)
				}
			},
		},
		{
			name: "multimodal_message_with_image",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("What's in this image?"),
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			expectError: false,
			validate: func(t *testing.T, msg openrouter.ChatCompletionMessage) {
				if len(msg.Content.Multi) != 2 {
					t.Errorf("Expected 2 content parts, got %d", len(msg.Content.Multi))
				}
				if msg.Content.Multi[0].Type != openrouter.ChatMessagePartTypeText {
					t.Errorf("Expected first part to be text")
				}
				if msg.Content.Multi[1].Type != openrouter.ChatMessagePartTypeImageURL {
					t.Errorf("Expected second part to be image")
				}
			},
		},
		{
			name: "image_with_unsupported_model",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
				},
			},
			expectError: true, // Will be tested with non-vision model
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			testClient := client
			if tt.name == "image_with_unsupported_model" {
				// Use a model that doesn't support vision
				testClient, _ = NewOpenRouterClient(ClientConfig{
					Provider: "openrouter",
					APIKey:   "test-key",
					Model:    "openai/gpt-3.5-turbo",
				})
				defer func() { _ = testClient.Close() }()
			}

			msg, err := testClient.convertMessage(tt.message)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.validate != nil {
				tt.validate(t, msg)
			}
		})
	}
}

// TestOpenRouterClient_convertResponse tests response conversion
func TestOpenRouterClient_convertResponse(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name     string
		response openrouter.ChatCompletionResponse
		validate func(t *testing.T, resp *ChatResponse)
	}{
		{
			name: "basic_response",
			response: openrouter.ChatCompletionResponse{
				ID:    "chatcmpl-123",
				Model: "openai/gpt-3.5-turbo",
				Choices: []openrouter.ChatCompletionChoice{
					{
						Index: 0,
						Message: openrouter.ChatCompletionMessage{
							Role:    "assistant",
							Content: openrouter.Content{Text: "Hello! How can I help you?"},
						},
						FinishReason: openrouter.FinishReasonStop,
					},
				},
				Usage: &openrouter.Usage{
					PromptTokens:     10,
					CompletionTokens: 8,
					TotalTokens:      18,
				},
			},
			validate: func(t *testing.T, resp *ChatResponse) {
				if resp.ID != "chatcmpl-123" {
					t.Errorf("Expected ID 'chatcmpl-123', got %s", resp.ID)
				}
				if resp.Model != "openai/gpt-3.5-turbo" {
					t.Errorf("Expected model 'openai/gpt-3.5-turbo', got %s", resp.Model)
				}
				if len(resp.Choices) != 1 {
					t.Errorf("Expected 1 choice, got %d", len(resp.Choices))
				}
				if resp.Choices[0].Message.GetText() != "Hello! How can I help you?" {
					t.Errorf("Expected message 'Hello! How can I help you?', got %s", resp.Choices[0].Message.GetText())
				}
				if resp.Usage.TotalTokens != 18 {
					t.Errorf("Expected total tokens 18, got %d", resp.Usage.TotalTokens)
				}
			},
		},
		{
			name: "response_with_tool_calls",
			response: openrouter.ChatCompletionResponse{
				ID:    "chatcmpl-456",
				Model: "openai/gpt-4",
				Choices: []openrouter.ChatCompletionChoice{
					{
						Index: 0,
						Message: openrouter.ChatCompletionMessage{
							Role:    "assistant",
							Content: openrouter.Content{Text: "I'll get the weather for you."},
							ToolCalls: []openrouter.ToolCall{
								{
									ID:   "call_123",
									Type: openrouter.ToolTypeFunction,
									Function: openrouter.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"location": "New York"}`,
									},
								},
							},
						},
						FinishReason: openrouter.FinishReasonToolCalls,
					},
				},
			},
			validate: func(t *testing.T, resp *ChatResponse) {
				if len(resp.Choices[0].Message.ToolCalls) != 1 {
					t.Errorf("Expected 1 tool call, got %d", len(resp.Choices[0].Message.ToolCalls))
				}
				if resp.Choices[0].Message.ToolCalls[0].ID != "call_123" {
					t.Errorf("Expected tool call ID 'call_123', got %s", resp.Choices[0].Message.ToolCalls[0].ID)
				}
				if resp.Choices[0].FinishReason != "tool_calls" {
					t.Errorf("Expected finish reason 'tool_calls', got %s", resp.Choices[0].FinishReason)
				}
			},
		},
		{
			name: "response_without_usage",
			response: openrouter.ChatCompletionResponse{
				ID:    "chatcmpl-789",
				Model: "openai/gpt-3.5-turbo",
				Choices: []openrouter.ChatCompletionChoice{
					{
						Index: 0,
						Message: openrouter.ChatCompletionMessage{
							Role:    "assistant",
							Content: openrouter.Content{Text: "Response without usage"},
						},
						FinishReason: openrouter.FinishReasonStop,
					},
				},
				Usage: nil,
			},
			validate: func(t *testing.T, resp *ChatResponse) {
				if resp.Usage.TotalTokens != 0 {
					t.Errorf("Expected zero usage when not provided, got %d", resp.Usage.TotalTokens)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resp := client.convertResponse(tt.response)

			if resp == nil {
				t.Error("Response should not be nil")
				return
			}

			if tt.validate != nil {
				tt.validate(t, resp)
			}
		})
	}
}

// TestOpenRouterClient_convertStreamResponse tests streaming response conversion
func TestOpenRouterClient_convertStreamResponse(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name     string
		response openrouter.ChatCompletionStreamResponse
		expected *StreamEvent
	}{
		{
			name: "delta_with_content",
			response: openrouter.ChatCompletionStreamResponse{
				Choices: []openrouter.ChatCompletionStreamChoice{
					{
						Index: 0,
						Delta: openrouter.ChatCompletionStreamChoiceDelta{
							Content: "Hello",
						},
					},
				},
			},
			expected: &StreamEvent{
				Type: "delta",
				Choice: &StreamChoice{
					Index: 0,
					Delta: &MessageDelta{
						Content: []MessageContent{NewTextContent("Hello")},
					},
				},
			},
		},
		{
			name: "delta_with_tool_call",
			response: openrouter.ChatCompletionStreamResponse{
				Choices: []openrouter.ChatCompletionStreamChoice{
					{
						Index: 0,
						Delta: openrouter.ChatCompletionStreamChoiceDelta{
							ToolCalls: []openrouter.ToolCall{
								{
									Index: intPtr(0),
									ID:    "call_123",
									Type:  openrouter.ToolTypeFunction,
									Function: openrouter.FunctionCall{
										Name:      "get_weather",
										Arguments: `{"location":`,
									},
								},
							},
						},
					},
				},
			},
			expected: &StreamEvent{
				Type: "delta",
				Choice: &StreamChoice{
					Index: 0,
					Delta: &MessageDelta{
						ToolCalls: []ToolCallDelta{
							{
								Index: 0,
								ID:    "call_123",
								Type:  "function",
								Function: &ToolCallFunctionDelta{
									Name:      "get_weather",
									Arguments: `{"location":`,
								},
							},
						},
					},
				},
			},
		},
		{
			name: "done_event",
			response: openrouter.ChatCompletionStreamResponse{
				Choices: []openrouter.ChatCompletionStreamChoice{
					{
						Index:        0,
						FinishReason: "stop",
					},
				},
			},
			expected: &StreamEvent{
				Type: "done",
				Choice: &StreamChoice{
					Index:        0,
					FinishReason: "stop",
				},
			},
		},
		{
			name: "empty_delta",
			response: openrouter.ChatCompletionStreamResponse{
				Choices: []openrouter.ChatCompletionStreamChoice{
					{
						Index: 0,
						Delta: openrouter.ChatCompletionStreamChoiceDelta{
							// Empty delta
						},
					},
				},
			},
			expected: nil, // Should return nil for empty deltas
		},
		{
			name:     "no_choices",
			response: openrouter.ChatCompletionStreamResponse{},
			expected: nil, // Should return nil for no choices
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertStreamResponse(tt.response)

			if tt.expected == nil {
				if result != nil {
					t.Errorf("Expected nil result, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Error("Expected non-nil result")
				return
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}

			if result.Choice == nil && tt.expected.Choice != nil {
				t.Error("Expected choice to be non-nil")
				return
			}

			if result.Choice != nil && tt.expected.Choice != nil {
				if result.Choice.Index != tt.expected.Choice.Index {
					t.Errorf("Expected index %d, got %d", tt.expected.Choice.Index, result.Choice.Index)
				}

				if result.Choice.FinishReason != tt.expected.Choice.FinishReason {
					t.Errorf("Expected finish reason %s, got %s", tt.expected.Choice.FinishReason, result.Choice.FinishReason)
				}

				// Compare delta content if present
				if tt.expected.Choice.Delta != nil && result.Choice.Delta != nil {
					if len(tt.expected.Choice.Delta.Content) > 0 && len(result.Choice.Delta.Content) > 0 {
						expectedText := tt.expected.Choice.Delta.Content[0].(*TextContent).GetText()
						actualText := result.Choice.Delta.Content[0].(*TextContent).GetText()
						if expectedText != actualText {
							t.Errorf("Expected content %s, got %s", expectedText, actualText)
						}
					}

					if len(tt.expected.Choice.Delta.ToolCalls) > 0 && len(result.Choice.Delta.ToolCalls) > 0 {
						expectedTC := tt.expected.Choice.Delta.ToolCalls[0]
						actualTC := result.Choice.Delta.ToolCalls[0]
						if expectedTC.ID != actualTC.ID {
							t.Errorf("Expected tool call ID %s, got %s", expectedTC.ID, actualTC.ID)
						}
					}
				}
			}
		})
	}
}

// TestOpenRouterClient_validateMultiModalContent tests multi-modal content validation
func TestOpenRouterClient_validateMultiModalContent(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		model       string
		content     []MessageContent
		expectError bool
		errorCode   string
	}{
		{
			name:  "text_only_content",
			model: "openai/gpt-3.5-turbo",
			content: []MessageContent{
				NewTextContent("Hello world"),
			},
			expectError: false,
		},
		{
			name:  "image_with_vision_model",
			model: "openai/gpt-4o",
			content: []MessageContent{
				NewTextContent("What's in this image?"),
				NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
			},
			expectError: false,
		},
		{
			name:  "image_with_non_vision_model",
			model: "openai/gpt-3.5-turbo",
			content: []MessageContent{
				NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
			},
			expectError: true,
			errorCode:   "unsupported_content_type",
		},
		{
			name:  "file_with_supporting_model",
			model: "openai/gpt-4",
			content: []MessageContent{
				NewFileContentFromBytes([]byte("test content"), "test.txt", "text/plain"),
			},
			expectError: false,
		},
		{
			name:  "file_with_non_supporting_model",
			model: "meta-llama/llama-3-8b",
			content: []MessageContent{
				NewFileContentFromBytes([]byte("test content"), "test.txt", "text/plain"),
			},
			expectError: true,
			errorCode:   "unsupported_content_type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewOpenRouterClient(ClientConfig{
				Provider: "openrouter",
				APIKey:   "test-key",
				Model:    tt.model,
			})
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer func() { _ = client.Close() }()

			err = client.validateMultiModalContent(tt.content)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
					return
				}

				if llmErr, ok := err.(*Error); ok {
					if llmErr.Code != tt.errorCode {
						t.Errorf("Expected error code %s, got %s", tt.errorCode, llmErr.Code)
					}
				} else {
					t.Errorf("Expected *Error type, got %T", err)
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestOpenRouterClient_validateToolSupport tests tool support validation
func TestOpenRouterClient_validateToolSupport(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		model       string
		tools       []Tool
		expectError bool
		errorCode   string
	}{
		{
			name:        "no_tools",
			model:       "openai/gpt-3.5-turbo",
			tools:       []Tool{},
			expectError: false,
		},
		{
			name:  "valid_tools_with_supporting_model",
			model: "openai/gpt-4",
			tools: []Tool{
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
			expectError: false,
		},
		{
			name:  "tools_with_non_supporting_model",
			model: "meta-llama/llama-3-8b",
			tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "test_function",
						Description: "Test function",
						Parameters: map[string]interface{}{
							"type": "object",
						},
					},
				},
			},
			expectError: true,
			errorCode:   "unsupported_feature",
		},
		{
			name:  "invalid_tool_definition",
			model: "openai/gpt-4",
			tools: []Tool{
				{
					Type: "function",
					Function: ToolFunction{
						Name:        "", // Empty name
						Description: "Test function",
						Parameters: map[string]interface{}{
							"type": "object",
						},
					},
				},
			},
			expectError: true,
			errorCode:   "invalid_tool_definition",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client, err := NewOpenRouterClient(ClientConfig{
				Provider: "openrouter",
				APIKey:   "test-key",
				Model:    tt.model,
			})
			if err != nil {
				t.Fatalf("Failed to create client: %v", err)
			}
			defer func() { _ = client.Close() }()

			err = client.validateToolSupport(tt.tools)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
					return
				}

				if llmErr, ok := err.(*Error); ok {
					if llmErr.Code != tt.errorCode {
						t.Errorf("Expected error code %s, got %s", tt.errorCode, llmErr.Code)
					}
				} else {
					t.Errorf("Expected *Error type, got %T", err)
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

// TestOpenRouterClient_convertError tests error conversion
func TestOpenRouterClient_convertError(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name     string
		inputErr error
		expected *Error
	}{
		{
			name:     "nil_error",
			inputErr: nil,
			expected: nil,
		},
		{
			name:     "generic_error",
			inputErr: errors.New("some error occurred"),
			expected: &Error{
				Code:    "openrouter_error",
				Message: "some error occurred",
				Type:    "api_error",
			},
		},
		{
			name:     "connection_refused_error",
			inputErr: errors.New("connection refused"),
			expected: &Error{
				Code:    "connection_error",
				Message: "connection refused",
				Type:    "network_error",
			},
		},
		{
			name:     "timeout_error",
			inputErr: errors.New("request timeout exceeded"),
			expected: &Error{
				Code:    "timeout_error",
				Message: "request timeout exceeded",
				Type:    "network_error",
			},
		},
		{
			name:     "context_canceled_error",
			inputErr: errors.New("context canceled"),
			expected: &Error{
				Code:    "request_canceled",
				Message: "context canceled",
				Type:    "network_error",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertError(tt.inputErr)

			if tt.expected == nil {
				if result != nil {
					t.Errorf("Expected nil error, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Error("Expected non-nil error")
				return
			}

			if result.Code != tt.expected.Code {
				t.Errorf("Expected code %s, got %s", tt.expected.Code, result.Code)
			}

			if result.Message != tt.expected.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Message, result.Message)
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}

			if result.StatusCode != tt.expected.StatusCode {
				t.Errorf("Expected status code %d, got %d", tt.expected.StatusCode, result.StatusCode)
			}
		})
	}
}

// TestConvertOpenRouterError tests comprehensive error conversion scenarios
func TestConvertOpenRouterError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		inputErr error
		expected *Error
	}{
		{
			name:     "nil_error",
			inputErr: nil,
			expected: nil,
		},
		{
			name: "api_error_401_invalid_key",
			inputErr: &openrouter.APIError{
				Code:           "invalid_api_key",
				Message:        "Invalid API key provided",
				HTTPStatusCode: 401,
			},
			expected: &Error{
				Code:       "invalid_api_key",
				Message:    "Invalid API key provided",
				Type:       "authentication_error",
				StatusCode: 401,
			},
		},
		{
			name: "api_error_401_missing_key",
			inputErr: &openrouter.APIError{
				Message:        "API key is missing",
				HTTPStatusCode: 401,
			},
			expected: &Error{
				Code:       "missing_api_key",
				Message:    "API key is missing",
				Type:       "authentication_error",
				StatusCode: 401,
			},
		},
		{
			name: "api_error_429_rate_limit",
			inputErr: &openrouter.APIError{
				Message:        "Rate limit exceeded. Please try again later",
				HTTPStatusCode: 429,
			},
			expected: &Error{
				Code:       "rate_limit_exceeded",
				Message:    "Rate limit exceeded. Please try again later",
				Type:       "rate_limit_error",
				StatusCode: 429,
			},
		},
		{
			name: "api_error_404_model_not_found",
			inputErr: &openrouter.APIError{
				Message:        "Model not found",
				HTTPStatusCode: 404,
			},
			expected: &Error{
				Code:       "model_not_found",
				Message:    "Model not found",
				Type:       "model_error",
				StatusCode: 404,
			},
		},
		{
			name: "api_error_400_token_limit",
			inputErr: &openrouter.APIError{
				Message:        "Token limit exceeded for this model",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "token_limit_exceeded",
				Message:    "Token limit exceeded for this model",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "api_error_400_context_length",
			inputErr: &openrouter.APIError{
				Message:        "Context length exceeded",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "context_length_exceeded",
				Message:    "Context length exceeded",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "api_error_400_content_filtered",
			inputErr: &openrouter.APIError{
				Message:        "Content violates content policy",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "content_filtered",
				Message:    "Content violates content policy",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "api_error_500_server_error",
			inputErr: &openrouter.APIError{
				Message:        "Internal server error",
				HTTPStatusCode: 500,
			},
			expected: &Error{
				Code:       "server_error",
				Message:    "Internal server error",
				Type:       "api_error",
				StatusCode: 500,
			},
		},
		{
			name: "request_error_401",
			inputErr: &openrouter.RequestError{
				HTTPStatus:     "401 Unauthorized",
				HTTPStatusCode: 401,
				Err:            errors.New("unauthorized"),
			},
			expected: &Error{
				Code:       "unauthorized",
				Message:    "error, status code: 401, status: 401 Unauthorized, message: unauthorized, body: ",
				Type:       "authentication_error",
				StatusCode: 401,
			},
		},
		{
			name: "request_error_429",
			inputErr: &openrouter.RequestError{
				HTTPStatus:     "429 Too Many Requests",
				HTTPStatusCode: 429,
				Err:            errors.New("rate limited"),
			},
			expected: &Error{
				Code:       "rate_limit_exceeded",
				Message:    "error, status code: 429, status: 429 Too Many Requests, message: rate limited, body: ",
				Type:       "rate_limit_error",
				StatusCode: 429,
			},
		},
		{
			name: "request_error_500",
			inputErr: &openrouter.RequestError{
				HTTPStatus:     "500 Internal Server Error",
				HTTPStatusCode: 500,
				Err:            errors.New("server error"),
			},
			expected: &Error{
				Code:       "server_error",
				Message:    "error, status code: 500, status: 500 Internal Server Error, message: server error, body: ",
				Type:       "api_error",
				StatusCode: 500,
			},
		},
		{
			name:     "network_connection_refused",
			inputErr: errors.New("connection refused"),
			expected: &Error{
				Code:    "connection_error",
				Message: "connection refused",
				Type:    "network_error",
			},
		},
		{
			name:     "network_no_such_host",
			inputErr: errors.New("no such host"),
			expected: &Error{
				Code:    "connection_error",
				Message: "no such host",
				Type:    "network_error",
			},
		},
		{
			name:     "network_timeout",
			inputErr: errors.New("request timeout"),
			expected: &Error{
				Code:    "timeout_error",
				Message: "request timeout",
				Type:    "network_error",
			},
		},
		{
			name:     "network_deadline_exceeded",
			inputErr: errors.New("context deadline exceeded"),
			expected: &Error{
				Code:    "timeout_error",
				Message: "context deadline exceeded",
				Type:    "network_error",
			},
		},
		{
			name:     "context_canceled",
			inputErr: errors.New("context canceled"),
			expected: &Error{
				Code:    "request_canceled",
				Message: "context canceled",
				Type:    "network_error",
			},
		},
		{
			name:     "tls_error",
			inputErr: errors.New("tls: certificate verification failed"),
			expected: &Error{
				Code:    "tls_error",
				Message: "tls: certificate verification failed",
				Type:    "network_error",
			},
		},
		{
			name:     "dns_error",
			inputErr: errors.New("dns lookup failed"),
			expected: &Error{
				Code:    "dns_error",
				Message: "dns lookup failed",
				Type:    "network_error",
			},
		},
		{
			name:     "generic_error_fallback",
			inputErr: errors.New("unknown error type"),
			expected: &Error{
				Code:    "openrouter_error",
				Message: "unknown error type",
				Type:    "api_error",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convertOpenRouterError(tt.inputErr)

			if tt.expected == nil {
				if result != nil {
					t.Errorf("Expected nil error, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Error("Expected non-nil error")
				return
			}

			if result.Code != tt.expected.Code {
				t.Errorf("Expected code %s, got %s", tt.expected.Code, result.Code)
			}

			if result.Message != tt.expected.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Message, result.Message)
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}

			if result.StatusCode != tt.expected.StatusCode {
				t.Errorf("Expected status code %d, got %d", tt.expected.StatusCode, result.StatusCode)
			}
		})
	}
}

// TestConvertAPIError tests APIError conversion specifically
func TestConvertAPIError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		apiErr   *openrouter.APIError
		expected *Error
	}{
		{
			name: "with_string_code",
			apiErr: &openrouter.APIError{
				Code:           "custom_error_code",
				Message:        "Custom error message",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "custom_error_code",
				Message:    "Custom error message",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "with_numeric_code",
			apiErr: &openrouter.APIError{
				Code:           42,
				Message:        "Numeric code error",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "bad_request",
				Message:    "Numeric code error",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "without_code",
			apiErr: &openrouter.APIError{
				Message:        "No code error",
				HTTPStatusCode: 403,
			},
			expected: &Error{
				Code:       "insufficient_permissions",
				Message:    "No code error",
				Type:       "authentication_error",
				StatusCode: 403,
			},
		},
		{
			name: "model_overloaded",
			apiErr: &openrouter.APIError{
				Message:        "Model is currently overloaded",
				HTTPStatusCode: 503,
			},
			expected: &Error{
				Code:       "model_overloaded",
				Message:    "Model is currently overloaded",
				Type:       "model_error",
				StatusCode: 503,
			},
		},
		{
			name: "model_not_supported",
			apiErr: &openrouter.APIError{
				Message:        "Model not supported for this operation",
				HTTPStatusCode: 400,
			},
			expected: &Error{
				Code:       "model_not_supported",
				Message:    "Model not supported for this operation",
				Type:       "model_error",
				StatusCode: 400,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convertAPIError(tt.apiErr)

			if result.Code != tt.expected.Code {
				t.Errorf("Expected code %s, got %s", tt.expected.Code, result.Code)
			}

			if result.Message != tt.expected.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Message, result.Message)
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}

			if result.StatusCode != tt.expected.StatusCode {
				t.Errorf("Expected status code %d, got %d", tt.expected.StatusCode, result.StatusCode)
			}
		})
	}
}

// TestConvertRequestError tests RequestError conversion specifically
func TestConvertRequestError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		reqErr   *openrouter.RequestError
		expected *Error
	}{
		{
			name: "with_http_status",
			reqErr: &openrouter.RequestError{
				HTTPStatus:     "400 Bad Request",
				HTTPStatusCode: 400,
				Err:            errors.New("bad request"),
			},
			expected: &Error{
				Code:       "bad_request",
				Message:    "error, status code: 400, status: 400 Bad Request, message: bad request, body: ",
				Type:       "validation_error",
				StatusCode: 400,
			},
		},
		{
			name: "without_http_status",
			reqErr: &openrouter.RequestError{
				HTTPStatusCode: 404,
				Err:            errors.New("not found"),
			},
			expected: &Error{
				Code:       "not_found",
				Message:    "error, status code: 404, status: , message: not found, body: ",
				Type:       "model_error",
				StatusCode: 404,
			},
		},
		{
			name: "client_error_range",
			reqErr: &openrouter.RequestError{
				HTTPStatus:     "422 Unprocessable Entity",
				HTTPStatusCode: 422,
				Err:            errors.New("validation failed"),
			},
			expected: &Error{
				Code:       "client_error",
				Message:    "error, status code: 422, status: 422 Unprocessable Entity, message: validation failed, body: ",
				Type:       "validation_error",
				StatusCode: 422,
			},
		},
		{
			name: "server_error_range",
			reqErr: &openrouter.RequestError{
				HTTPStatus:     "502 Bad Gateway",
				HTTPStatusCode: 502,
				Err:            errors.New("bad gateway"),
			},
			expected: &Error{
				Code:       "server_error",
				Message:    "error, status code: 502, status: 502 Bad Gateway, message: bad gateway, body: ",
				Type:       "api_error",
				StatusCode: 502,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convertRequestError(tt.reqErr)

			if result.Code != tt.expected.Code {
				t.Errorf("Expected code %s, got %s", tt.expected.Code, result.Code)
			}

			if result.Message != tt.expected.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Message, result.Message)
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}

			if result.StatusCode != tt.expected.StatusCode {
				t.Errorf("Expected status code %d, got %d", tt.expected.StatusCode, result.StatusCode)
			}
		})
	}
}

// TestConvertCommonError tests common error conversion
func TestConvertCommonError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		inputErr error
		expected *Error
	}{
		{
			name:     "connection_refused",
			inputErr: errors.New("dial tcp: connection refused"),
			expected: &Error{
				Code:    "connection_error",
				Message: "dial tcp: connection refused",
				Type:    "network_error",
			},
		},
		{
			name:     "no_such_host",
			inputErr: errors.New("dial tcp: lookup api.openrouter.ai: no such host"),
			expected: &Error{
				Code:    "connection_error",
				Message: "dial tcp: lookup api.openrouter.ai: no such host",
				Type:    "network_error",
			},
		},
		{
			name:     "network_unreachable",
			inputErr: errors.New("dial tcp: network is unreachable"),
			expected: &Error{
				Code:    "connection_error",
				Message: "dial tcp: network is unreachable",
				Type:    "network_error",
			},
		},
		{
			name:     "timeout",
			inputErr: errors.New("dial tcp: i/o timeout"),
			expected: &Error{
				Code:    "timeout_error",
				Message: "dial tcp: i/o timeout",
				Type:    "network_error",
			},
		},
		{
			name:     "deadline_exceeded",
			inputErr: errors.New("context deadline exceeded"),
			expected: &Error{
				Code:    "timeout_error",
				Message: "context deadline exceeded",
				Type:    "network_error",
			},
		},
		{
			name:     "context_canceled",
			inputErr: errors.New("context canceled"),
			expected: &Error{
				Code:    "request_canceled",
				Message: "context canceled",
				Type:    "network_error",
			},
		},
		{
			name:     "tls_error",
			inputErr: errors.New("x509: certificate signed by unknown authority"),
			expected: &Error{
				Code:    "tls_error",
				Message: "x509: certificate signed by unknown authority",
				Type:    "network_error",
			},
		},
		{
			name:     "dns_error",
			inputErr: errors.New("lookup api.openrouter.ai: dns server failure"),
			expected: &Error{
				Code:    "dns_error",
				Message: "lookup api.openrouter.ai: dns server failure",
				Type:    "network_error",
			},
		},
		{
			name:     "unrecognized_error",
			inputErr: errors.New("some random error"),
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convertCommonError(tt.inputErr)

			if tt.expected == nil {
				if result != nil {
					t.Errorf("Expected nil error, got %+v", result)
				}
				return
			}

			if result == nil {
				t.Error("Expected non-nil error")
				return
			}

			if result.Code != tt.expected.Code {
				t.Errorf("Expected code %s, got %s", tt.expected.Code, result.Code)
			}

			if result.Message != tt.expected.Message {
				t.Errorf("Expected message %s, got %s", tt.expected.Message, result.Message)
			}

			if result.Type != tt.expected.Type {
				t.Errorf("Expected type %s, got %s", tt.expected.Type, result.Type)
			}
		})
	}
}

// TestOpenRouterClient_convertImageContent tests image content conversion
func TestOpenRouterClient_convertImageContent(t *testing.T) {
	t.Parallel()

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-4o",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	tests := []struct {
		name        string
		image       *ImageContent
		expectError bool
		validate    func(t *testing.T, part openrouter.ChatMessagePart)
	}{
		{
			name:        "image_with_url",
			image:       NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg"),
			expectError: false,
			validate: func(t *testing.T, part openrouter.ChatMessagePart) {
				if part.Type != openrouter.ChatMessagePartTypeImageURL {
					t.Errorf("Expected type ImageURL, got %s", part.Type)
				}
				if part.ImageURL.URL != "https://example.com/image.jpg" {
					t.Errorf("Expected URL 'https://example.com/image.jpg', got %s", part.ImageURL.URL)
				}
			},
		},
		{
			name:        "image_with_data",
			image:       NewImageContentFromBytes([]byte("fake-image-data"), "image/png"),
			expectError: false,
			validate: func(t *testing.T, part openrouter.ChatMessagePart) {
				if part.Type != openrouter.ChatMessagePartTypeImageURL {
					t.Errorf("Expected type ImageURL, got %s", part.Type)
				}
				if !strings.HasPrefix(part.ImageURL.URL, "data:image/png;base64,") {
					t.Errorf("Expected data URL with PNG prefix, got %s", part.ImageURL.URL)
				}
			},
		},
		{
			name: "image_without_data_or_url",
			image: &ImageContent{
				MimeType: "image/jpeg",
			},
			expectError: true,
		},
		{
			name:        "image_with_unsupported_mime_type_data",
			image:       NewImageContentFromBytes([]byte("fake-image-data"), "image/bmp"),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			part, err := client.convertImageContent(tt.image)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if tt.validate != nil {
				tt.validate(t, part)
			}
		})
	}
}

// TestOpenRouterClient_StreamingMockData tests streaming functionality with mock data
func TestOpenRouterClient_StreamingMockData(t *testing.T) {
	t.Parallel()

	// This test simulates streaming behavior without making actual API calls
	// by testing the convertStreamResponse method with various mock responses

	client, err := NewOpenRouterClient(ClientConfig{
		Provider: "openrouter",
		APIKey:   "test-key",
		Model:    "openai/gpt-3.5-turbo",
	})
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer func() { _ = client.Close() }()

	// Test streaming response sequence
	streamResponses := []openrouter.ChatCompletionStreamResponse{
		{
			Choices: []openrouter.ChatCompletionStreamChoice{
				{
					Index: 0,
					Delta: openrouter.ChatCompletionStreamChoiceDelta{
						Content: "Hello",
					},
				},
			},
		},
		{
			Choices: []openrouter.ChatCompletionStreamChoice{
				{
					Index: 0,
					Delta: openrouter.ChatCompletionStreamChoiceDelta{
						Content: " world",
					},
				},
			},
		},
		{
			Choices: []openrouter.ChatCompletionStreamChoice{
				{
					Index: 0,
					Delta: openrouter.ChatCompletionStreamChoiceDelta{
						Content: "!",
					},
				},
			},
		},
		{
			Choices: []openrouter.ChatCompletionStreamChoice{
				{
					Index:        0,
					FinishReason: "stop",
				},
			},
		},
	}

	var events []StreamEvent
	for _, resp := range streamResponses {
		if event := client.convertStreamResponse(resp); event != nil {
			events = append(events, *event)
		}
	}

	// Verify we got the expected events
	if len(events) != 4 {
		t.Errorf("Expected 4 events, got %d", len(events))
	}

	// Check delta events
	for i := 0; i < 3; i++ {
		if !events[i].IsDelta() {
			t.Errorf("Event %d should be a delta event", i)
		}
	}

	// Check done event
	if !events[3].IsDone() {
		t.Error("Last event should be a done event")
	}

	// Verify content accumulation
	expectedContent := []string{"Hello", " world", "!"}
	for i, expected := range expectedContent {
		if events[i].Choice.Delta == nil || len(events[i].Choice.Delta.Content) == 0 {
			t.Errorf("Event %d should have content", i)
			continue
		}
		actual := events[i].Choice.Delta.Content[0].(*TextContent).GetText()
		if actual != expected {
			t.Errorf("Event %d: expected content %s, got %s", i, expected, actual)
		}
	}

	// Verify finish reason
	if events[3].Choice.FinishReason != "stop" {
		t.Errorf("Expected finish reason 'stop', got %s", events[3].Choice.FinishReason)
	}
}

// Helper functions for tests

func floatPtr(f float32) *float32 {
	return &f
}

func intPtr(i int) *int {
	return &i
}
