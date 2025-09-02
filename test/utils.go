package test

import (
	"embed"
	"io/fs"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/factory"
	"github.com/inercia/go-llm/pkg/llm"
)

//go:embed fixtures/*
var fixtures embed.FS

// createTestClient creates a client using environment configuration
func createTestClient(t *testing.T) llm.Client {
	t.Helper()

	factory := factory.New()
	config := llm.GetLLMFromEnv()

	client, err := factory.CreateClient(config)
	require.NoError(t, err, "Failed to create LLM client")
	require.NotNil(t, client, "Client should not be nil")

	// Log which provider we're using
	info := client.GetModelInfo()
	t.Logf("🤖 Using %s provider with model %s", info.Provider, info.Name)

	return client
}

// createTestClientWithTimeout creates a client with custom timeout
func createTestClientWithTimeout(t *testing.T, timeout time.Duration) llm.Client {
	t.Helper()

	factory := factory.New()
	config := llm.GetLLMFromEnv()
	config.Timeout = timeout

	client, err := factory.CreateClient(config)
	require.NoError(t, err, "Failed to create LLM client with timeout")
	require.NotNil(t, client, "Client should not be nil")

	return client
}

// getFixtureImages returns a list of fixture images available for testing
func getFixtureImages(t *testing.T) []string {
	t.Helper()

	var images []string
	err := fs.WalkDir(fixtures, "fixtures", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			// Only include image files
			filename := strings.TrimPrefix(path, "fixtures/")
			if isImageFile(filename) {
				images = append(images, filename)
			}
		}
		return nil
	})
	require.NoError(t, err)
	return images
}

// isImageFile checks if a filename is an image based on extension
func isImageFile(filename string) bool {
	lower := strings.ToLower(filename)
	return strings.HasSuffix(lower, ".jpg") ||
		strings.HasSuffix(lower, ".jpeg") ||
		strings.HasSuffix(lower, ".png") ||
		strings.HasSuffix(lower, ".gif") ||
		strings.HasSuffix(lower, ".webp")
}

// getMimeType determines MIME type from filename
func getMimeType(filename string) string {
	lower := strings.ToLower(filename)
	if strings.HasSuffix(lower, ".png") {
		return "image/png"
	}
	if strings.HasSuffix(lower, ".gif") {
		return "image/gif"
	}
	if strings.HasSuffix(lower, ".webp") {
		return "image/webp"
	}
	// Default to JPEG for .jpg, .jpeg and unknown
	return "image/jpeg"
}

// skipIfNoProvider skips the test if no LLM provider is available
func skipIfNoProvider(t *testing.T, client llm.Client) {
	t.Helper()

	info := client.GetModelInfo()

	// Check if it's actually the mock provider (which means no real provider was available)
	if info.Provider == "mock" && (info.Name == "fallback" || info.Name == "no-provider") {
		t.Skip("No LLM provider available - set OPENAI_API_KEY, GEMINI_API_KEY, or start Ollama server")
	}
}

// requireVisionSupport skips the test if the provider doesn't support vision
func requireVisionSupport(t *testing.T, client llm.Client) {
	t.Helper()

	info := client.GetModelInfo()
	if !info.SupportsVision {
		t.Skipf("Provider %s model %s doesn't support vision", info.Provider, info.Name)
	}
}

// requireToolSupport skips the test if the provider doesn't support tools
func requireToolSupport(t *testing.T, client llm.Client) {
	t.Helper()

	info := client.GetModelInfo()
	if !info.SupportsTools {
		t.Skipf("Provider %s model %s doesn't support tools", info.Provider, info.Name)
	}
}
