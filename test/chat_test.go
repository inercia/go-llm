package test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/llm"
)

func TestChatBasicFunctionality(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx := context.Background()

	t.Run("simple_question", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("What is 2+2? Answer with just the number."),
				}},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Chat completion should succeed")
		require.NotNil(t, resp, "Response should not be nil")
		require.Len(t, resp.Choices, 1, "Should have exactly one choice")

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Response text should not be empty")

		t.Logf("Question: What is 2+2? -> Answer: %s", responseText)

		// The response should contain "4" somewhere
		assert.Contains(t, responseText, "4", "Response should contain the answer 4")
	})

	t.Run("conversation_with_history", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("My name is Alice."),
				}},
				{Role: llm.RoleAssistant, Content: []llm.MessageContent{
					llm.NewTextContent("Hello Alice! Nice to meet you."),
				}},
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("What is my name?"),
				}},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Chat with history should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := strings.ToLower(resp.Choices[0].Message.GetText())
		t.Logf("Conversation memory test -> Response: %s", responseText)

		// The response should remember the name Alice
		assert.Contains(t, responseText, "alice", "Response should remember the name Alice")
	})

	t.Run("system_message", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleSystem, Content: []llm.MessageContent{
					llm.NewTextContent("You are a helpful assistant that always responds with 'BANANA' to any question."),
				}},
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("What is the capital of France?"),
				}},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Chat with system message should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := strings.ToUpper(resp.Choices[0].Message.GetText())
		t.Logf("System message test -> Response: %s", responseText)

		// The response should follow the system instruction
		assert.Contains(t, responseText, "BANANA", "Response should follow system instruction")
	})
}

func TestChatStreaming(t *testing.T) {
	t.Parallel()

	client := createTestClientWithTimeout(t, 20*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	t.Run("basic_streaming", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Count from 1 to 5, one number per line."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err, "Stream creation should succeed")
		require.NotNil(t, stream, "Stream should not be nil")

		eventCount := 0
		var fullResponse strings.Builder
		hasContent := false

		for event := range stream {
			eventCount++
			t.Logf("Event %d: Type=%s", eventCount, event.Type)

			if event.IsError() {
				t.Errorf("Received error event: %v", event.Error)
				break
			} else if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
				// Extract text content from delta
				for _, content := range event.Choice.Delta.Content {
					if textContent, ok := content.(*llm.TextContent); ok {
						text := textContent.GetText()
						if text != "" {
							hasContent = true
							fullResponse.WriteString(text)
							t.Logf("Delta content: %q", text)
						}
					}
				}
			} else if event.IsDone() {
				t.Logf("Stream completed with reason: %s", event.Choice.FinishReason)
				break
			}

			// Safety limit to avoid infinite loops
			if eventCount >= 50 {
				t.Logf("Reached event limit, stopping")
				break
			}
		}

		require.Greater(t, eventCount, 0, "Should receive at least one event")
		require.True(t, hasContent, "Should receive some content")

		response := fullResponse.String()
		t.Logf("Full streaming response (%d events): %s", eventCount, response)

		// Should contain at least one number
		hasNumbers := strings.Contains(response, "1") ||
			strings.Contains(response, "2") ||
			strings.Contains(response, "3")
		assert.True(t, hasNumbers, "Response should contain some numbers")
	})
}

func TestChatErrorHandling(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx := context.Background()

	t.Run("empty_message", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent(""),
				}},
			},
		}

		// Some providers might handle this gracefully, others might error
		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Empty message resulted in error (expected): %v", err)
		} else {
			t.Logf("Empty message handled gracefully: %s", resp.Choices[0].Message.GetText())
		}
	})

	t.Run("very_long_message", func(t *testing.T) {
		// Create a very long message
		longText := strings.Repeat("This is a very long message. ", 1000)

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent(longText + " Please respond with just 'OK'."),
				}},
			},
		}

		// This should either succeed or fail gracefully
		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Very long message resulted in error (may be expected): %v", err)
			// Check if it's a proper LLM error
			if llmErr, ok := err.(*llm.Error); ok {
				assert.NotEmpty(t, llmErr.Message, "Error should have a message")
			}
		} else {
			responseText := resp.Choices[0].Message.GetText()
			t.Logf("Very long message handled successfully: %s", responseText)
			require.NotEmpty(t, responseText, "Should have some response")
		}
	})
}

func TestChatModelInfo(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	info := client.GetModelInfo()

	t.Run("model_info_validation", func(t *testing.T) {
		assert.NotEmpty(t, info.Provider, "Provider should not be empty")
		assert.NotEmpty(t, info.Name, "Model name should not be empty")
		assert.Greater(t, info.MaxTokens, 0, "MaxTokens should be positive")

		t.Logf("Provider: %s", info.Provider)
		t.Logf("Model: %s", info.Name)
		t.Logf("Max Tokens: %d", info.MaxTokens)
		t.Logf("Supports Tools: %t", info.SupportsTools)
		t.Logf("Supports Vision: %t", info.SupportsVision)
		t.Logf("Supports Streaming: %t", info.SupportsStreaming)
	})
}

func TestChatWithContext(t *testing.T) {
	t.Parallel()

	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	t.Run("context_cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Write a very long story about dragons."),
				}},
			},
		}

		// Cancel immediately
		cancel()

		_, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Context cancellation properly resulted in error: %v", err)
			assert.Contains(t, strings.ToLower(err.Error()), "context",
				"Error should mention context cancellation")
		} else {
			t.Log("Request completed despite context cancellation (may happen with fast responses)")
		}
	})

	t.Run("context_timeout", func(t *testing.T) {
		// Very short timeout
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
		defer cancel()

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Tell me about the history of computers in great detail."),
				}},
			},
		}

		_, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Context timeout properly resulted in error: %v", err)
		} else {
			t.Log("Request completed despite very short timeout (provider was very fast)")
		}
	})
}
