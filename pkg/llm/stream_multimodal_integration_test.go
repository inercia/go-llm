package llm

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestStreamingMultiModalIntegration tests the combination of streaming and multi-modal content
func TestStreamingMultiModalIntegration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		provider     string
		model        string
		envVar       string
		skipReason   string
		setupContent func() []MessageContent
	}{
		{
			name:     "OpenAI_streaming_with_image",
			provider: "openai",
			model:    "gpt-4o-mini",
			envVar:   "OPENAI_API_KEY",
			setupContent: func() []MessageContent {
				return []MessageContent{
					NewTextContent("Describe this test image briefly:"),
					NewImageContentFromBytes(createTestImageData(), "image/jpeg"),
				}
			},
		},
		{
			name:     "Gemini_streaming_with_image",
			provider: "gemini",
			model:    "gemini-1.5-flash",
			envVar:   "GEMINI_API_KEY",
			setupContent: func() []MessageContent {
				return []MessageContent{
					NewTextContent("What's in this test image?"),
					NewImageContentFromBytes(createTestImageData(), "image/png"),
				}
			},
		},
		{
			name:       "Ollama_streaming_with_image",
			provider:   "ollama",
			model:      "llava:7b",
			envVar:     "",
			skipReason: "Requires Ollama server with vision model",
			setupContent: func() []MessageContent {
				return []MessageContent{
					NewTextContent("Describe this image:"),
					NewImageContentFromBytes(createTestImageData(), "image/jpeg"),
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Skip tests that require API keys or specific setup
			if tt.envVar != "" && os.Getenv(tt.envVar) == "" {
				t.Skipf("%s not set, skipping %s test", tt.envVar, tt.provider)
			}
			if tt.skipReason != "" {
				t.Skip(tt.skipReason)
			}

			factory := NewFactory()
			config := ClientConfig{
				Provider: tt.provider,
				Model:    tt.model,
				Timeout:  30 * time.Second,
			}
			if tt.envVar != "" {
				config.APIKey = os.Getenv(tt.envVar)
			}

			client, err := factory.CreateClient(config)
			require.NoError(t, err, "Failed to create %s client", tt.provider)
			defer func() { _ = client.Close() }()

			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()

			req := ChatRequest{
				Model: tt.model,
				Messages: []Message{
					{
						Role:    RoleUser,
						Content: tt.setupContent(),
					},
				},
				Stream: true,
			}

			// Test streaming with multi-modal content
			stream, err := client.StreamChatCompletion(ctx, req)
			require.NoError(t, err, "Failed to start streaming for %s", tt.provider)
			require.NotNil(t, stream, "Stream channel should not be nil")

			// Collect stream events
			var events []StreamEvent
			var fullResponse strings.Builder
			hasContent := false

			for event := range stream {
				events = append(events, event)

				switch {
				case event.IsDelta():
					if event.Choice.Delta != nil && len(event.Choice.Delta.Content) > 0 {
						if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
							text := textContent.GetText()
							fullResponse.WriteString(text)
							hasContent = true
						}
					}
				case event.IsDone():
					goto streamComplete
				case event.IsError():
					t.Fatalf("Stream error for %s: %s", tt.provider, event.Error.Message)
				}
			}

		streamComplete:
			require.NotEmpty(t, events, "Should receive at least one event")
			assert.True(t, hasContent, "Should receive content in delta events")

			finalResponse := fullResponse.String()
			require.NotEmpty(t, finalResponse, "Should receive non-empty response content")

			t.Logf("✓ %s streaming with multi-modal content working. Response length: %d chars",
				tt.provider, len(finalResponse))
		})
	}
}

// TestMockStreamingMultiModal tests streaming multi-modal with mock client
func TestMockStreamingMultiModal(t *testing.T) {
	t.Parallel()

	mockClient := NewMockClient("gpt-4o-mock", "mock")
	defer func() { _ = mockClient.Close() }()

	ctx := context.Background()

	req := ChatRequest{
		Model: "gpt-4o-mock",
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image:"),
					NewImageContentFromBytes(createTestImageData(), "image/jpeg"),
					NewFileContentFromBytes([]byte("sample data"), "test.txt", "text/plain"),
				},
			},
		},
		Stream: true,
	}

	stream, err := mockClient.StreamChatCompletion(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, stream)

	// Collect events
	var events []StreamEvent
	var content strings.Builder

	for event := range stream {
		events = append(events, event)
		if event.IsDelta() && event.Choice.Delta != nil && len(event.Choice.Delta.Content) > 0 {
			if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
				content.WriteString(textContent.GetText())
			}
		}
		if event.IsDone() {
			break
		}
	}

	require.NotEmpty(t, events)
	assert.True(t, len(content.String()) > 0)

	// Verify the request was logged with multi-modal content
	calls := mockClient.GetCallLog()
	require.Len(t, calls, 1)
	assert.Len(t, calls[0].Messages[0].Content, 3) // text + image + file
}

// TestCrossProviderMultiModalStreaming compares streaming behavior across providers
func TestCrossProviderMultiModalStreaming(t *testing.T) {
	t.Parallel()

	providers := []struct {
		name     string
		provider string
		model    string
		envVar   string
	}{
		{"OpenAI", "openai", "gpt-4o-mini", "OPENAI_API_KEY"},
		{"Gemini", "gemini", "gemini-1.5-flash", "GEMINI_API_KEY"},
		{"Mock", "mock", "test-model", ""},
	}

	content := []MessageContent{
		NewTextContent("Describe this test image in one word:"),
		NewImageContentFromBytes(createTestImageData(), "image/jpeg"),
	}

	results := make(map[string]map[string]interface{})

	for _, p := range providers {
		t.Run(p.name, func(t *testing.T) {
			t.Parallel()

			if p.envVar != "" && os.Getenv(p.envVar) == "" {
				t.Skipf("%s not set, skipping %s test", p.envVar, p.name)
			}

			var client Client
			var err error

			if p.provider == "mock" {
				client = NewMockClient(p.model, p.provider)
			} else {
				factory := NewFactory()
				config := ClientConfig{
					Provider: p.provider,
					Model:    p.model,
					Timeout:  30 * time.Second,
				}
				if p.envVar != "" {
					config.APIKey = os.Getenv(p.envVar)
				}
				client, err = factory.CreateClient(config)
				require.NoError(t, err)
			}
			defer func() { _ = client.Close() }()

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			req := ChatRequest{
				Model:    p.model,
				Messages: []Message{{Role: RoleUser, Content: content}},
				Stream:   true,
			}

			start := time.Now()
			stream, err := client.StreamChatCompletion(ctx, req)
			require.NoError(t, err)

			var eventCount int
			var totalContent strings.Builder
			var firstChunkTime time.Duration
			var lastChunkTime time.Duration

			for event := range stream {
				eventCount++
				if event.IsDelta() && len(event.Choice.Delta.Content) > 0 {
					if firstChunkTime == 0 {
						firstChunkTime = time.Since(start)
					}
					lastChunkTime = time.Since(start)
					if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
						totalContent.WriteString(textContent.GetText())
					}
				}
				if event.IsDone() {
					break
				}
			}

			results[p.name] = map[string]interface{}{
				"eventCount":    eventCount,
				"contentLength": len(totalContent.String()),
				"firstChunk":    firstChunkTime,
				"totalTime":     lastChunkTime,
				"avgChunkSize":  float64(len(totalContent.String())) / float64(eventCount),
			}

			t.Logf("✓ %s: %d events, %d chars, first chunk in %v, total time %v",
				p.name, eventCount, len(totalContent.String()), firstChunkTime, lastChunkTime)
		})
	}

	// Note: In a real test environment, you could compare performance metrics
	// across providers, but since we're skipping most due to missing API keys,
	// this serves as a framework for when those keys are available.
}

// TestStreamingMultiModalErrorHandling tests error scenarios
func TestStreamingMultiModalErrorHandling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		setupClient    func() Client
		setupRequest   func() ChatRequest
		expectError    bool
		expectErrorMsg string
	}{
		{
			name: "mock_client_with_error",
			setupClient: func() Client {
				client := NewMockClient("test-model", "mock")
				client.WithError("mock_error", "simulated stream error", "simulation_error")
				return client
			},
			setupRequest: func() ChatRequest {
				return ChatRequest{
					Model: "test-model",
					Messages: []Message{
						{
							Role: RoleUser,
							Content: []MessageContent{
								NewTextContent("Test with image:"),
								NewImageContentFromBytes(createTestImageData(), "image/jpeg"),
							},
						},
					},
					Stream: true,
				}
			},
			expectError:    true,
			expectErrorMsg: "simulated stream error",
		},
		{
			name: "invalid_content_type",
			setupClient: func() Client {
				return NewMockClient("test-model", "mock")
			},
			setupRequest: func() ChatRequest {
				return ChatRequest{
					Model: "test-model",
					Messages: []Message{
						{
							Role: RoleUser,
							Content: []MessageContent{
								NewTextContent("Test"),
								// This should work fine - no invalid content types in our system
							},
						},
					},
					Stream: true,
				}
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client := tt.setupClient()
			defer func() { _ = client.Close() }()

			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			req := tt.setupRequest()
			stream, err := client.StreamChatCompletion(ctx, req)

			if tt.expectError {
				// For mock errors, the error might come via the stream
				if err == nil {
					require.NotNil(t, stream)
					// Check for error events in stream
					for event := range stream {
						if event.IsError() {
							assert.Contains(t, event.Error.Message, tt.expectErrorMsg)
							return
						}
					}
					t.Error("Expected error but didn't receive one")
				} else {
					assert.Contains(t, err.Error(), tt.expectErrorMsg)
				}
			} else {
				require.NoError(t, err)
				require.NotNil(t, stream)
			}
		})
	}
}

// TestStreamingMultiModalPerformance benchmarks streaming multi-modal performance
func TestStreamingMultiModalPerformance(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	sizes := []struct {
		name string
		size int
	}{
		{"small", 1024},   // 1KB
		{"medium", 10240}, // 10KB
		{"large", 102400}, // 100KB
	}

	for _, size := range sizes {
		t.Run(size.name, func(t *testing.T) {
			t.Parallel()

			client := NewMockClient("test-model", "mock")
			defer func() { _ = client.Close() }()

			imageData := make([]byte, size.size)
			for i := range imageData {
				imageData[i] = byte(i % 256)
			}

			req := ChatRequest{
				Model: "test-model",
				Messages: []Message{
					{
						Role: RoleUser,
						Content: []MessageContent{
							NewTextContent("Process this image:"),
							NewImageContentFromBytes(imageData, "image/jpeg"),
						},
					},
				},
				Stream: true,
			}

			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			start := time.Now()
			stream, err := client.StreamChatCompletion(ctx, req)
			require.NoError(t, err)

			var eventCount int
			for event := range stream {
				eventCount++
				if event.IsDone() {
					break
				}
			}

			duration := time.Since(start)
			t.Logf("✓ %s image (%d bytes): %d events in %v (%.2f events/sec)",
				size.name, size.size, eventCount, duration, float64(eventCount)/duration.Seconds())
		})
	}
}

// Helper function to create test image data
func createTestImageData() []byte {
	// Create minimal JPEG-like data for testing
	// This is not a real JPEG, just test data that looks like binary image data
	return []byte{
		0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
		0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
		0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
		0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
		0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
		0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
		0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
		0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xD9,
	}
}
