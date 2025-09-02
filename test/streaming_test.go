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

func TestStreamingBasicFunctionality(t *testing.T) {
	client := createTestClientWithTimeout(t, 15*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	t.Run("simple_streaming_response", func(t *testing.T) {
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
		deltaCount := 0
		var fullResponse strings.Builder
		hasContent := false
		finishReason := ""

		for event := range stream {
			eventCount++
			t.Logf("Event %d: Type=%s", eventCount, event.Type)

			if event.IsError() {
				t.Errorf("Received error event: %v", event.Error)
				break
			} else if event.IsDelta() {
				deltaCount++
				if event.Choice != nil && event.Choice.Delta != nil {
					// Extract text content from delta
					for _, content := range event.Choice.Delta.Content {
						if textContent, ok := content.(*llm.TextContent); ok {
							text := textContent.GetText()
							if text != "" {
								hasContent = true
								fullResponse.WriteString(text)
								t.Logf("Delta %d: %q", deltaCount, text)
							}
						}
					}
				}
			} else if event.IsDone() {
				if event.Choice != nil {
					finishReason = event.Choice.FinishReason
				}
				t.Logf("Stream completed with reason: %s", finishReason)
				break
			}

			// Safety limit to avoid infinite loops
			if eventCount >= 100 {
				t.Logf("Reached event limit, stopping")
				break
			}
		}

		require.Greater(t, eventCount, 0, "Should receive at least one event")
		require.Greater(t, deltaCount, 0, "Should receive at least one delta event")
		require.True(t, hasContent, "Should receive some content")

		response := fullResponse.String()
		t.Logf("Full streaming response (%d events, %d deltas): %s", eventCount, deltaCount, response)

		// Should contain at least one number from the counting task
		hasNumbers := strings.Contains(response, "1") ||
			strings.Contains(response, "2") ||
			strings.Contains(response, "3") ||
			strings.Contains(response, "4") ||
			strings.Contains(response, "5")
		assert.True(t, hasNumbers, "Response should contain some numbers from counting")

		// Should have a proper finish reason
		assert.NotEmpty(t, finishReason, "Should have a finish reason")
	})

	t.Run("streaming_conversation", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Hello, please introduce yourself briefly."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err, "Conversation stream should succeed")
		require.NotNil(t, stream)

		var response strings.Builder
		eventCount := 0

		for event := range stream {
			eventCount++

			if event.IsError() {
				t.Errorf("Stream error: %v", event.Error)
				break
			} else if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
				for _, content := range event.Choice.Delta.Content {
					if textContent, ok := content.(*llm.TextContent); ok {
						response.WriteString(textContent.GetText())
					}
				}
			} else if event.IsDone() {
				break
			}

			if eventCount >= 50 {
				break
			}
		}

		responseText := response.String()
		t.Logf("Conversation response: %s", responseText)

		// Should have some introduction-like content
		lowerResponse := strings.ToLower(responseText)
		hasIntroduction := strings.Contains(lowerResponse, "hello") ||
			strings.Contains(lowerResponse, "assistant") ||
			strings.Contains(lowerResponse, "help") ||
			strings.Contains(lowerResponse, "ai") ||
			len(responseText) > 10 // At least some response

		assert.True(t, hasIntroduction, "Response should contain introduction-like content")
	})
}

func TestStreamingPerformance(t *testing.T) {
	client := createTestClientWithTimeout(t, 20*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	t.Run("time_to_first_token", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Say 'Hello' and then count to 10."),
				}},
			},
			Stream: true,
		}

		startTime := time.Now()
		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, stream)

		var firstTokenTime time.Time
		var lastTokenTime time.Time
		tokenCount := 0

		for event := range stream {
			if event.IsError() {
				t.Errorf("Stream error: %v", event.Error)
				break
			} else if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
				for _, content := range event.Choice.Delta.Content {
					if textContent, ok := content.(*llm.TextContent); ok {
						text := textContent.GetText()
						if text != "" {
							tokenCount++
							if firstTokenTime.IsZero() {
								firstTokenTime = time.Now()
							}
							lastTokenTime = time.Now()
						}
					}
				}
			} else if event.IsDone() {
				break
			}
		}

		if !firstTokenTime.IsZero() {
			timeToFirstToken := firstTokenTime.Sub(startTime)
			totalStreamTime := lastTokenTime.Sub(firstTokenTime)

			t.Logf("⏱️ Performance metrics:")
			t.Logf("   Time to first token: %v", timeToFirstToken)
			t.Logf("   Total streaming time: %v", totalStreamTime)
			t.Logf("   Token count: %d", tokenCount)
			if tokenCount > 0 && totalStreamTime > 0 {
				tokensPerSecond := float64(tokenCount) / totalStreamTime.Seconds()
				t.Logf("   Tokens per second: %.2f", tokensPerSecond)
			}

			// Performance assertions (reasonable expectations)
			assert.Less(t, timeToFirstToken, 10*time.Second, "First token should arrive within reasonable time")
			assert.Greater(t, tokenCount, 0, "Should receive some tokens")
		}
	})
}

func TestStreamingWithLongContent(t *testing.T) {
	client := createTestClientWithTimeout(t, 30*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	t.Run("long_response_streaming", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Write a short story about a robot learning to paint. Keep it under 200 words."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, stream)

		var fullStory strings.Builder
		eventCount := 0
		wordCount := 0

		for event := range stream {
			eventCount++

			if event.IsError() {
				t.Errorf("Stream error: %v", event.Error)
				break
			} else if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
				for _, content := range event.Choice.Delta.Content {
					if textContent, ok := content.(*llm.TextContent); ok {
						text := textContent.GetText()
						fullStory.WriteString(text)
						// Rough word counting
						wordCount += len(strings.Fields(text))
					}
				}
			} else if event.IsDone() {
				t.Logf("Story completed with %d events", eventCount)
				break
			}

			// Safety limit for very long responses
			if eventCount >= 200 {
				t.Log("Reached event limit for long content")
				break
			}
		}

		story := fullStory.String()
		t.Logf("Generated story (%d events, ~%d words): %s", eventCount, wordCount, story)

		// Should have substantial content
		assert.Greater(t, len(story), 50, "Story should have substantial content")
		assert.Greater(t, wordCount, 10, "Story should have multiple words")

		// Should be story-like content
		lowerStory := strings.ToLower(story)
		hasStoryElements := strings.Contains(lowerStory, "robot") ||
			strings.Contains(lowerStory, "paint") ||
			strings.Contains(lowerStory, "art") ||
			len(story) > 100 // At least some substantial content

		assert.True(t, hasStoryElements, "Response should contain story elements")
	})
}

func TestStreamingErrorHandling(t *testing.T) {
	client := createTestClientWithTimeout(t, 10*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	t.Run("context_cancellation_during_streaming", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel() // Ensure cancel is always called

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Write a very long essay about artificial intelligence history."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, stream)

		eventCount := 0

		// Process a few events then cancel
		for event := range stream {
			eventCount++
			t.Logf("Event %d before cancellation: %s", eventCount, event.Type)

			if eventCount >= 2 {
				t.Log("Cancelling context...")
				cancel()
			}

			if event.IsError() {
				t.Logf("Received error after cancellation: %v", event.Error)
				break
			} else if event.IsDone() {
				break
			}

			// Safety limit
			if eventCount >= 20 {
				break
			}
		}

		t.Logf("Processed %d events before/after cancellation", eventCount)
	})

	t.Run("timeout_during_streaming", func(t *testing.T) {
		// Very short timeout
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Write a detailed analysis of quantum computing."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Stream creation failed with timeout (expected): %v", err)
			return
		}

		require.NotNil(t, stream)

		eventCount := 0
		for event := range stream {
			eventCount++

			if event.IsError() {
				t.Logf("Received timeout error in stream: %v", event.Error)
				break
			} else if event.IsDone() {
				break
			}

			// Shouldn't get too many events with short timeout
			if eventCount >= 5 {
				t.Log("Got more events than expected with short timeout")
				break
			}
		}

		t.Logf("Processed %d events with short timeout", eventCount)
	})
}

func TestStreamingEventTypes(t *testing.T) {
	client := createTestClientWithTimeout(t, 10*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	t.Run("event_type_validation", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Say 'Hello World' and stop."),
				}},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err)
		require.NotNil(t, stream)

		eventTypes := make(map[string]int)

		for event := range stream {
			eventTypes[event.Type]++

			// Validate event structure
			if event.IsDelta() {
				assert.Equal(t, "delta", event.Type, "Delta event should have correct type")
				assert.NotNil(t, event.Choice, "Delta event should have choice")
			} else if event.IsDone() {
				assert.Equal(t, "done", event.Type, "Done event should have correct type")
				assert.NotNil(t, event.Choice, "Done event should have choice")
				assert.NotEmpty(t, event.Choice.FinishReason, "Done event should have finish reason")
				break
			} else if event.IsError() {
				assert.Equal(t, "error", event.Type, "Error event should have correct type")
				assert.NotNil(t, event.Error, "Error event should have error")
				break
			}

			// Safety limit
			if len(eventTypes) > 10 {
				break
			}
		}

		t.Logf("Event types received: %+v", eventTypes)

		// Should have received at least delta events
		assert.Greater(t, eventTypes["delta"], 0, "Should receive delta events")

		// Should end with done or error
		hasFinalEvent := eventTypes["done"] > 0 || eventTypes["error"] > 0
		assert.True(t, hasFinalEvent, "Should end with done or error event")
	})
}
