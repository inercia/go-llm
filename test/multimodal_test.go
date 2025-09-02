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

func TestMultiModalWithFixtureImages(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireVisionSupport(t, client)

	// Get available fixture images
	images := getFixtureImages(t)
	require.Greater(t, len(images), 0, "Should have fixture images available")

	ctx := context.Background()

	// Test each image type
	imageTests := []struct {
		filename    string
		description string
		expectation string // What we expect to be described
	}{
		{"car.jpeg", "a car", "car"},
		{"tree.jpeg", "a tree", "tree"},
		{"test-jpeg.jpg", "test image", ""},
		{"test-png.png", "test image", ""},
		{"colorful-pattern.png", "colorful pattern", "pattern"},
		{"geometric-shapes.png", "geometric shapes", "shape"},
		{"simple-text.png", "simple text", "text"},
	}

	for _, tt := range imageTests {
		t.Run(tt.description, func(t *testing.T) {
			// Check if this image exists in fixtures
			found := false
			for _, img := range images {
				if img == tt.filename {
					found = true
					break
				}
			}
			if !found {
				t.Skipf("Image %s not found in fixtures", tt.filename)
				return
			}

			// Load image data
			imageData, err := fixtures.ReadFile("fixtures/" + tt.filename)
			require.NoError(t, err, "Failed to read image file")
			require.Greater(t, len(imageData), 0, "Image data should not be empty")

			mimeType := getMimeType(tt.filename)

			// Create multimodal request
			req := llm.ChatRequest{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []llm.MessageContent{
							llm.NewTextContent("Please describe what you see in this image briefly."),
							llm.NewImageContentFromBytes(imageData, mimeType),
						},
					},
				},
			}

			// Send request
			resp, err := client.ChatCompletion(ctx, req)
			require.NoError(t, err, "ChatCompletion should succeed")
			require.NotNil(t, resp, "Response should not be nil")
			require.Len(t, resp.Choices, 1, "Should have exactly one choice")

			// Verify response content
			responseText := resp.Choices[0].Message.GetText()
			require.NotEmpty(t, responseText, "Response text should not be empty")

			// Check if response makes sense for the image
			if tt.expectation != "" {
				lowerResponse := strings.ToLower(responseText)
				lowerExpectation := strings.ToLower(tt.expectation)
				assert.Contains(t, lowerResponse, lowerExpectation,
					"Response should mention expected content: %s", tt.expectation)
			}

			// Log for debugging
			t.Logf("Image: %s (%d bytes) -> Response: %s", tt.filename, len(imageData), responseText)
		})
	}
}

func TestMultiModalStreamingWithImages(t *testing.T) {
	client := createTestClientWithTimeout(t, 15*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireVisionSupport(t, client)

	// Get available images
	images := getFixtureImages(t)
	require.Greater(t, len(images), 0, "Should have fixture images available")

	// Use the first available image for streaming test
	testImage := images[0]
	imageData, err := fixtures.ReadFile("fixtures/" + testImage)
	require.NoError(t, err, "Failed to read test image")

	mimeType := getMimeType(testImage)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	t.Run("basic_streaming_with_image", func(t *testing.T) {
		// Create streaming multimodal request
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("Describe what you see in this image. Stream your response."),
						llm.NewImageContentFromBytes(imageData, mimeType),
					},
				},
			},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err, "StreamChatCompletion should succeed")
		require.NotNil(t, stream, "Stream should not be nil")

		// Consume events
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
						}
					}
				}
			} else if event.IsDone() {
				t.Logf("Stream completed with reason: %s", event.Choice.FinishReason)
				break
			}

			// Don't consume all events to avoid long test times
			if eventCount >= 20 {
				t.Log("Reached event limit")
				break
			}
		}

		require.Greater(t, eventCount, 0, "Should receive at least one streaming event")

		response := fullResponse.String()
		t.Logf("Streaming multimodal test with %s (%d bytes): %d events, response: %q",
			testImage, len(imageData), eventCount, response)

		// Should have received some content
		assert.True(t, hasContent, "Should receive some content in streaming response")
	})
}

func TestMultiModalWithMultipleImages(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireVisionSupport(t, client)

	// Get available images
	images := getFixtureImages(t)
	require.Greater(t, len(images), 1, "Need at least 2 images for this test")

	ctx := context.Background()

	t.Run("compare_two_images", func(t *testing.T) {
		// Load first two images
		var imageContents []llm.MessageContent
		imageContents = append(imageContents, llm.NewTextContent("Compare these two images and describe their differences:"))

		for i, imageName := range images[:2] { // Use first 2 images
			imageData, err := fixtures.ReadFile("fixtures/" + imageName)
			require.NoError(t, err, "Failed to read image %s", imageName)

			mimeType := getMimeType(imageName)

			imageContents = append(imageContents,
				llm.NewImageContentFromBytes(imageData, mimeType))

			t.Logf("Added image %d: %s (%d bytes)", i+1, imageName, len(imageData))
		}

		// Create request with multiple images
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role:    llm.RoleUser,
					Content: imageContents,
				},
			},
		}

		// Send request
		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Multi-image request should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Response should not be empty")

		// Response should mention comparison or differences
		lowerResponse := strings.ToLower(responseText)
		hasComparison := strings.Contains(lowerResponse, "differ") ||
			strings.Contains(lowerResponse, "compar") ||
			strings.Contains(lowerResponse, "both") ||
			strings.Contains(lowerResponse, "similar") ||
			strings.Contains(lowerResponse, "contrast")

		assert.True(t, hasComparison, "Response should indicate image comparison")

		t.Logf("Multi-image comparison response: %s", responseText)
	})
}

func TestMultiModalWithFileContent(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx := context.Background()

	t.Run("analyze_json_file", func(t *testing.T) {
		// Create test JSON data
		jsonData := []byte(`{
			"name": "multimodal_test",
			"type": "integration_test", 
			"version": "1.0",
			"features": ["vision", "text", "files"],
			"metadata": {
				"created": "2024-01-01",
				"purpose": "testing multimodal capabilities"
			}
		}`)

		// Create request with file content
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("Analyze this JSON file and explain its structure briefly:"),
						llm.NewFileContentFromBytes(jsonData, "test.json", "application/json"),
					},
				},
			},
		}

		// Send request
		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "File content request should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Response should not be empty")

		// Response should mention JSON or file analysis
		lowerResponse := strings.ToLower(responseText)
		hasFileAnalysis := strings.Contains(lowerResponse, "json") ||
			strings.Contains(lowerResponse, "structure") ||
			strings.Contains(lowerResponse, "object") ||
			strings.Contains(lowerResponse, "multimodal")

		assert.True(t, hasFileAnalysis, "Response should analyze the JSON file")

		t.Logf("JSON file analysis (%d bytes): %s", len(jsonData), responseText)
	})

	t.Run("mixed_content_types", func(t *testing.T) {
		// Get a test image
		images := getFixtureImages(t)
		if len(images) == 0 {
			t.Skip("No images available for mixed content test")
		}

		imageData, err := fixtures.ReadFile("fixtures/" + images[0])
		require.NoError(t, err, "Failed to read image")

		textData := []byte("This is a text document that describes the image.")

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("I'm providing both an image and a text file. Compare what you see in the image with what the text describes:"),
						llm.NewImageContentFromBytes(imageData, getMimeType(images[0])),
						llm.NewFileContentFromBytes(textData, "description.txt", "text/plain"),
					},
				},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Mixed content request should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Response should not be empty")

		t.Logf("Mixed content response: %s", responseText)
	})
}

func TestMultiModalErrorHandling(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)

	ctx := context.Background()

	t.Run("empty_image_data", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("Describe this image:"),
						llm.NewImageContentFromBytes([]byte{}, "image/jpeg"), // Empty data
					},
				},
			},
		}

		// This might error or be handled gracefully depending on provider
		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Empty image properly caused error: %v", err)
			// Should be a proper LLM error
			if llmErr, ok := err.(*llm.Error); ok {
				assert.NotEmpty(t, llmErr.Message, "Error should have a message")
			}
		} else {
			t.Logf("Empty image handled gracefully: %s", resp.Choices[0].Message.GetText())
			require.NotNil(t, resp)
		}
	})

	t.Run("invalid_image_format", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("Describe this image:"),
						llm.NewImageContentFromBytes([]byte("this-is-not-an-image"), "image/jpeg"),
					},
				},
			},
		}

		// This might error or be handled gracefully
		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Invalid image properly caused error: %v", err)
		} else {
			t.Logf("Invalid image handled gracefully: %s", resp.Choices[0].Message.GetText())
			require.NotNil(t, resp)
		}
	})

	t.Run("unsupported_file_type", func(t *testing.T) {
		// Try with a binary file type
		binaryData := []byte{0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []llm.MessageContent{
						llm.NewTextContent("Analyze this binary file:"),
						llm.NewFileContentFromBytes(binaryData, "test.bin", "application/octet-stream"),
					},
				},
			},
		}

		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Unsupported file type properly caused error: %v", err)
		} else {
			t.Logf("Unsupported file type handled: %s", resp.Choices[0].Message.GetText())
			require.NotNil(t, resp)
		}
	})
}

func TestMultiModalPerformance(t *testing.T) {
	client := createTestClientWithTimeout(t, 30*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireVisionSupport(t, client)

	// Get available images
	images := getFixtureImages(t)
	if len(images) == 0 {
		t.Skip("No images available for performance test")
	}

	ctx := context.Background()

	t.Run("image_size_performance", func(t *testing.T) {
		// Test different sized images if available
		for _, imageName := range images[:3] { // Test up to 3 images
			t.Run(imageName, func(t *testing.T) {
				imageData, err := fixtures.ReadFile("fixtures/" + imageName)
				require.NoError(t, err)

				start := time.Now()

				req := llm.ChatRequest{
					Messages: []llm.Message{
						{
							Role: llm.RoleUser,
							Content: []llm.MessageContent{
								llm.NewTextContent("Briefly describe this image."),
								llm.NewImageContentFromBytes(imageData, getMimeType(imageName)),
							},
						},
					},
				}

				resp, err := client.ChatCompletion(ctx, req)
				duration := time.Since(start)

				require.NoError(t, err, "Image processing should succeed")
				require.NotNil(t, resp)

				responseText := resp.Choices[0].Message.GetText()
				t.Logf("✓ %s (%d bytes): processed in %v -> %s",
					imageName, len(imageData), duration, responseText[:min(100, len(responseText))])

				// Performance should be reasonable (under 30 seconds)
				assert.Less(t, duration, 30*time.Second, "Image processing should complete in reasonable time")
			})
		}
	})
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
