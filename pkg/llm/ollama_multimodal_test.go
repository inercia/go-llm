package llm

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestOllamaClient_MultiModal_ModelSupport(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llava:13b"}

	tests := []struct {
		name        string
		request     ChatRequest
		expectError bool
	}{
		{
			name: "text_only_request",
			request: ChatRequest{
				Messages: []Message{
					NewTextMessage(RoleUser, "Hello world"),
				},
			},
			expectError: false,
		},
		{
			name: "image_content_request",
			request: ChatRequest{
				Messages: []Message{
					{
						Role: RoleUser,
						Content: []MessageContent{
							NewTextContent("Describe this image:"),
							NewImageContentFromBytes([]byte("test-image-data"), "image/jpeg"),
						},
					},
				},
			},
			expectError: false,
		},
		{
			name: "multiple_images_request",
			request: ChatRequest{
				Messages: []Message{
					{
						Role: RoleUser,
						Content: []MessageContent{
							NewTextContent("Compare these images:"),
							NewImageContentFromBytes([]byte("image1-data"), "image/jpeg"),
							NewImageContentFromBytes([]byte("image2-data"), "image/png"),
						},
					},
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertToOllamaRequest(tt.request)

			// Basic structure validation
			if len(result.Messages) == 0 {
				t.Error("Expected at least one message")
			}

			if result.Model != client.model {
				t.Errorf("Expected model %s, got %s", client.model, result.Model)
			}
		})
	}
}

func TestOllamaClient_ConvertToOllamaRequest_MultiModal(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llava:13b"}

	// Create test data
	testImageData := []byte("fake-jpeg-data")
	testFileData := []byte("file content")

	req := ChatRequest{
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Analyze this image and file:"),
					NewImageContentFromBytes(testImageData, "image/jpeg"),
					NewFileContentFromBytes(testFileData, "data.txt", "text/plain"),
				},
			},
		},
	}

	result := client.convertToOllamaRequest(req)

	// Verify basic structure
	if len(result.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(result.Messages))
	}

	message := result.Messages[0]
	if message.Role != "user" {
		t.Errorf("Expected user role, got %s", message.Role)
	}

	// Verify content contains text parts
	if !strings.Contains(message.Content, "Analyze this image and file:") {
		t.Error("Expected original text content in message")
	}

	if !strings.Contains(message.Content, "[Image attached]") {
		t.Error("Expected image placeholder in message content")
	}

	if !strings.Contains(message.Content, "[File: data.txt") {
		t.Error("Expected file description in message content")
	}

	// Verify images array
	if len(result.Images) != 1 {
		t.Fatalf("Expected 1 image, got %d", len(result.Images))
	}

	expectedImageData := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(testImageData)
	if result.Images[0] != expectedImageData {
		t.Error("Expected correct image data encoding")
	}
}

func TestOllamaClient_ConvertToOllamaRequest_ImageFromURL(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llava:13b"}

	req := ChatRequest{
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image:"),
					NewImageContentFromURL("https://example.com/test.jpg", "image/jpeg"),
				},
			},
		},
	}

	result := client.convertToOllamaRequest(req)

	// Verify URL-based images are handled as placeholders
	message := result.Messages[0]
	if !strings.Contains(message.Content, "[Image: https://example.com/test.jpg]") {
		t.Error("Expected URL placeholder in message content")
	}

	// Should not have any images in the images array for URL-based content
	if len(result.Images) != 0 {
		t.Error("Expected no images for URL-based content")
	}
}

func TestOllamaClient_ConvertToOllamaRequest_TextOnly(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llama2:7b"}

	req := ChatRequest{
		Messages: []Message{
			NewTextMessage(RoleSystem, "You are a helpful assistant"),
			NewTextMessage(RoleUser, "Hello"),
			NewTextMessage(RoleAssistant, "Hi there!"),
		},
	}

	result := client.convertToOllamaRequest(req)

	if len(result.Messages) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(result.Messages))
	}

	// Verify role conversion
	expectedRoles := []string{"system", "user", "assistant"}
	for i, msg := range result.Messages {
		if msg.Role != expectedRoles[i] {
			t.Errorf("Expected role %s, got %s", expectedRoles[i], msg.Role)
		}
	}

	// Verify no images for text-only content
	if len(result.Images) != 0 {
		t.Error("Expected no images for text-only content")
	}
}

func TestOllamaClient_ConvertToOllamaRequest_EmptyContent(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llama2:7b"}

	req := ChatRequest{
		Messages: []Message{
			{
				Role:    RoleUser,
				Content: []MessageContent{},
			},
		},
	}

	result := client.convertToOllamaRequest(req)

	if len(result.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(result.Messages))
	}

	if result.Messages[0].Content != "" {
		t.Error("Expected empty content for empty message")
	}
}

func TestOllamaClient_ConvertToOllamaRequest_WithOptions(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llama2:7b"}

	temperature := float32(0.7)
	maxTokens := 1000
	topP := float32(0.9)

	req := ChatRequest{
		Messages: []Message{
			NewTextMessage(RoleUser, "Hello"),
		},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
		TopP:        &topP,
	}

	result := client.convertToOllamaRequest(req)

	if result.Options == nil {
		t.Fatal("Expected options to be set")
	}

	if result.Options.Temperature == nil || *result.Options.Temperature != 0.7 {
		t.Error("Expected temperature to be set to 0.7")
	}

	if result.Options.NumPredict == nil || *result.Options.NumPredict != 1000 {
		t.Error("Expected num_predict to be set to 1000")
	}

	if result.Options.TopP == nil || *result.Options.TopP != 0.9 {
		t.Error("Expected top_p to be set to 0.9")
	}
}

func TestOllamaClient_ConvertToOllamaRequest_MixedContent(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llava:13b"}

	req := ChatRequest{
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Process this data:"),
					NewImageContentFromBytes([]byte("image-data"), "image/png"),
					NewFileContentFromBytes([]byte(`{"key": "value"}`), "data.json", "application/json"),
					NewTextContent("And provide analysis."),
				},
			},
		},
	}

	result := client.convertToOllamaRequest(req)

	message := result.Messages[0]

	// Verify all text content is included
	if !strings.Contains(message.Content, "Process this data:") {
		t.Error("Expected first text part")
	}
	if !strings.Contains(message.Content, "And provide analysis.") {
		t.Error("Expected second text part")
	}

	// Verify image placeholder
	if !strings.Contains(message.Content, "[Image attached]") {
		t.Error("Expected image placeholder")
	}

	// Verify file description
	if !strings.Contains(message.Content, "[File: data.json") {
		t.Error("Expected file description")
	}

	// Verify image data
	if len(result.Images) != 1 {
		t.Error("Expected one image")
	}
}

func TestOllamaClient_ConvertRoleToOllama(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{}

	tests := []struct {
		input    MessageRole
		expected string
	}{
		{RoleSystem, "system"},
		{RoleUser, "user"},
		{RoleAssistant, "assistant"},
		{RoleTool, "assistant"}, // Ollama converts tool to assistant
	}

	for _, tt := range tests {
		t.Run(string(tt.input), func(t *testing.T) {
			result := client.convertRoleToOllama(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

func TestOllamaClient_ConvertRoleFromOllama(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{}

	tests := []struct {
		input    string
		expected MessageRole
	}{
		{"system", RoleSystem},
		{"user", RoleUser},
		{"assistant", RoleAssistant},
		{"unknown", RoleUser}, // Default fallback
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := client.convertRoleFromOllama(tt.input)
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}

// Benchmark tests for Ollama multi-modal performance
func BenchmarkOllamaClient_ConvertToOllamaRequest_TextOnly(b *testing.B) {
	client := &OllamaClient{model: "llama2:7b"}
	req := ChatRequest{
		Messages: []Message{
			NewTextMessage(RoleUser, "Hello world"),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertToOllamaRequest(req)
	}
}

func BenchmarkOllamaClient_ConvertToOllamaRequest_MultiModal(b *testing.B) {
	client := &OllamaClient{model: "llava:13b"}
	testData := []byte("test-image-data")

	req := ChatRequest{
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this:"),
					NewImageContentFromBytes(testData, "image/jpeg"),
				},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertToOllamaRequest(req)
	}
}

// Test edge cases and error scenarios
func TestOllamaClient_MultiModal_EdgeCases(t *testing.T) {
	t.Parallel()

	client := &OllamaClient{model: "llava:13b"}

	tests := []struct {
		name     string
		message  Message
		expected int // expected number of images
	}{
		{
			name: "large_image_data",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Analyze this large image:"),
					NewImageContentFromBytes(make([]byte, 1024*1024), "image/jpeg"), // 1MB image
				},
			},
			expected: 1,
		},
		{
			name: "multiple_images",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Compare these images:"),
					NewImageContentFromBytes([]byte("image1"), "image/jpeg"),
					NewImageContentFromBytes([]byte("image2"), "image/png"),
					NewImageContentFromBytes([]byte("image3"), "image/gif"),
				},
			},
			expected: 3,
		},
		{
			name: "empty_image_data",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Process this:"),
					NewImageContentFromBytes([]byte{}, "image/jpeg"), // Empty image
				},
			},
			expected: 0, // Empty image data results in no image being added
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			req := ChatRequest{Messages: []Message{tt.message}}
			result := client.convertToOllamaRequest(req)

			if len(result.Images) != tt.expected {
				t.Errorf("Expected %d images, got %d", tt.expected, len(result.Images))
			}

			// Verify all images have proper data URL format
			for i, img := range result.Images {
				if !strings.HasPrefix(img, "data:") {
					t.Errorf("Image %d doesn't have data URL format: %s", i, img)
				}
				if !strings.Contains(img, "base64,") {
					t.Errorf("Image %d doesn't contain base64 marker: %s", i, img)
				}
			}
		})
	}
}
