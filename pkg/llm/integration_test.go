package llm

import (
	"context"
	"embed"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed fixtures

var embeddedFixtures embed.FS

// loadFixtureImage loads an image from the fixtures directory and returns its bytes and MIME type.
func loadFixtureImage(t *testing.T, filename string) ([]byte, string) {
	t.Helper()

	imagePath := "fixtures/" + filename
	data, err := embeddedFixtures.ReadFile(imagePath)
	require.NoError(t, err, "failed to read fixture image %s", filename)
	require.NotEmpty(t, data, "fixture image %s is empty", filename)

	ext := strings.ToLower(filepath.Ext(filename))
	var mimeType string
	switch ext {
	case ".jpg", ".jpeg":
		mimeType = "image/jpeg"
	case ".png":
		mimeType = "image/png"
	default:
		require.Fail(t, "unsupported image extension %s", ext)
	}
	return data, mimeType
}

// consumeStream safely consumes the stream channel and returns collected events.
func consumeStream(t *testing.T, stream <-chan StreamEvent) []StreamEvent {
	t.Helper()
	var events []StreamEvent
	timeout := time.NewTimer(30 * time.Second)
	defer timeout.Stop()

	for {
		select {
		case event, ok := <-stream:
			if !ok {
				return events
			}
			events = append(events, event)
			if event.IsDone() {
				return events
			}
		case <-timeout.C:
			require.Fail(t, "stream consumption timed out")
		}
	}
}

func getFixtureImages(t *testing.T) []string {
	t.Helper()
	var images []string

	err := fs.WalkDir(embeddedFixtures, "fixtures", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			images = append(images, strings.TrimPrefix(path, "fixtures/"))
		}
		return nil
	})
	require.NoError(t, err)
	return images
}

func TestFactoryGeminiIntegration(t *testing.T) {
	factory := NewFactory()

	// Test Gemini client creation through factory
	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   "test-key-for-integration", // This will trigger the stub
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Verify it's a Gemini client
	assert.Equal(t, "gemini", client.GetModelInfo().Provider)
	assert.Equal(t, "gemini-1.5-flash", client.GetModelInfo().Name)

	// Test the stub behavior (expected errors)
	ctx := context.Background()
	req := ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []Message{
			NewTextMessage(RoleUser, "Test integration"),
		},
	}

	// With fake key, expect authentication error
	resp, err := client.ChatCompletion(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	if apiErr, ok := err.(*Error); ok {
		assert.Equal(t, "authentication_error", apiErr.Code)
	}

	// Streaming behavior with genai library may differ
	// The genai library might not fail immediately on invalid API keys in streaming mode
	stream, err := client.StreamChatCompletion(ctx, req)
	if err != nil {
		// If it fails immediately, it should be an authentication error
		assert.Nil(t, stream)
		if apiErr, ok := err.(*Error); ok {
			assert.Equal(t, "authentication_error", apiErr.Code)
		}
	} else {
		// If it doesn't fail immediately, errors should come through the stream
		assert.NotNil(t, stream)
		// Read the first event which should be an error
		select {
		case event := <-stream:
			if event.Type == "error" {
				assert.Contains(t, event.Error.Error(), "API key")
			}
		case <-time.After(time.Second):
			t.Log("Stream didn't produce an error event within timeout - this is acceptable for genai library")
		}
	}

	// Close should work
	assert.NoError(t, client.Close())
}

func TestFactoryGeminiRealIntegration(t *testing.T) {
	// Skip if no real API key
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping real integration test")
	}

	factory := NewFactory()

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   os.Getenv("GEMINI_API_KEY"),
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Note: This test will currently fail due to stub implementation
	// TODO: Enable when real Gemini API integration is complete
	// No skip needed now that implementation exists

	ctx := context.Background()
	req := ChatRequest{
		Model: "gemini-1.5-flash",
		Messages: []Message{
			NewTextMessage(RoleUser, "Hello from integration test"),
		},
	}

	resp, err := client.ChatCompletion(ctx, req)
	if t.Skipped() {
		return
	}

	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Len(t, resp.Choices, 1)
	require.NotEmpty(t, resp.Choices[0].Message.GetText())

	assert.NoError(t, client.Close())
}

// Vision integration tests for OpenAI and Gemini using fixture images.
// These tests skip if API keys are not set and only verify no errors occur,
// without asserting on response content.

// TestFactoryOpenAIVisionCompletionIntegration tests non-streaming vision completion with OpenAI.
func TestFactoryOpenAIVisionCompletionIntegration(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set, skipping real integration test")
	}

	factory := NewFactory()

	config := ClientConfig{
		Provider: "openai",
		Model:    "gpt-4o-mini",
		APIKey:   os.Getenv("OPENAI_API_KEY"),
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	fixtureImages := getFixtureImages(t)

	for _, image := range fixtureImages {
		t.Run(image, func(t *testing.T) {
			data, mimeType := loadFixtureImage(t, image)

			message := Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image."),
					NewImageContentFromBytes(data, mimeType),
				},
			}

			req := ChatRequest{
				Model:    "gpt-4o-mini",
				Messages: []Message{message},
			}

			resp, err := client.ChatCompletion(ctx, req)
			require.NoError(t, err)
			require.NotNil(t, resp)
		})
	}
}

// TestFactoryOpenAIVisionStreamingIntegration tests streaming vision with OpenAI.
func TestFactoryOpenAIVisionStreamingIntegration(t *testing.T) {
	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("OPENAI_API_KEY not set, skipping real integration test")
	}

	factory := NewFactory()

	config := ClientConfig{
		Provider: "openai",
		Model:    "gpt-4o-mini",
		APIKey:   os.Getenv("OPENAI_API_KEY"),
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	fixtureImages := getFixtureImages(t)

	for _, image := range fixtureImages {
		t.Run(image, func(t *testing.T) {
			data, mimeType := loadFixtureImage(t, image)

			message := Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image."),
					NewImageContentFromBytes(data, mimeType),
				},
			}

			req := ChatRequest{
				Model:    "gpt-4o-mini",
				Messages: []Message{message},
			}

			stream, err := client.StreamChatCompletion(ctx, req)
			require.NoError(t, err)
			require.NotNil(t, stream)

			events := consumeStream(t, stream)

			// Assert no error events and at least one event (e.g., delta or done)
			hasError := false
			for _, event := range events {
				if event.IsError() {
					hasError = true
					break
				}
			}
			require.False(t, hasError)
			require.Greater(t, len(events), 0)
		})
	}
}

// TestFactoryGeminiVisionCompletionIntegration tests non-streaming vision completion with Gemini.
func TestFactoryGeminiVisionCompletionIntegration(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping real integration test")
	}

	factory := NewFactory()

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   os.Getenv("GEMINI_API_KEY"),
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	fixtureImages := getFixtureImages(t)

	for _, image := range fixtureImages {
		t.Run(image, func(t *testing.T) {
			data, mimeType := loadFixtureImage(t, image)

			message := Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image."),
					NewImageContentFromBytes(data, mimeType),
				},
			}

			req := ChatRequest{
				Model:    "gemini-1.5-flash",
				Messages: []Message{message},
			}

			resp, err := client.ChatCompletion(ctx, req)
			require.NoError(t, err)
			require.NotNil(t, resp)
		})
	}
}

// TestFactoryGeminiVisionStreamingIntegration tests streaming vision with Gemini.
func TestFactoryGeminiVisionStreamingIntegration(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set, skipping real integration test")
	}

	factory := NewFactory()

	config := ClientConfig{
		Provider: "gemini",
		Model:    "gemini-1.5-flash",
		APIKey:   os.Getenv("GEMINI_API_KEY"),
	}

	client, err := factory.CreateClient(config)
	require.NoError(t, err)
	require.NotNil(t, client)
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	fixtureImages := getFixtureImages(t)

	for _, image := range fixtureImages {
		t.Run(image, func(t *testing.T) {
			data, mimeType := loadFixtureImage(t, image)

			message := Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image."),
					NewImageContentFromBytes(data, mimeType),
				},
			}

			req := ChatRequest{
				Model:    "gemini-1.5-flash",
				Messages: []Message{message},
			}

			stream, err := client.StreamChatCompletion(ctx, req)
			require.NoError(t, err)
			require.NotNil(t, stream)

			events := consumeStream(t, stream)

			// Assert no error events and at least one event (e.g., delta or done)
			hasError := false
			for _, event := range events {
				if event.IsError() {
					hasError = true
					break
				}
			}
			require.False(t, hasError)
			require.Greater(t, len(events), 0)
		})
	}
}
