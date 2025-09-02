package ollama

import (
	"encoding/base64"
	"strings"
	"testing"

	"github.com/inercia/go-llm/pkg/llm"
)

const DefaultOllamaMultimodalModel = "llava:13b"

func TestClient_MultiModal_ModelSupport(t *testing.T) {
	t.Parallel()

	client := &Client{model: DefaultOllamaMultimodalModel}

	tests := []struct {
		name        string
		request     llm.ChatRequest
		expectError bool
	}{
		{
			name: "text_only_request",
			request: llm.ChatRequest{
				Messages: []llm.Message{
					llm.NewTextMessage(llm.RoleUser, "Hello world"),
				},
			},
			expectError: false,
		},
		{
			name: "image_content_request",
			request: llm.ChatRequest{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []llm.MessageContent{
							llm.NewTextContent("Describe this image:"),
							llm.NewImageContentFromBytes([]byte("test-image-data"), "image/jpeg"),
						},
					},
				},
			},
			expectError: false,
		},
		{
			name: "multiple_images_request",
			request: llm.ChatRequest{
				Messages: []llm.Message{
					{
						Role: llm.RoleUser,
						Content: []llm.MessageContent{
							llm.NewTextContent("Compare these images:"),
							llm.NewImageContentFromBytes([]byte("image1-data"), "image/jpeg"),
							llm.NewImageContentFromBytes([]byte("image2-data"), "image/png"),
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

func TestClient_ConvertToOllamaRequest_MultiModal(t *testing.T) {
	t.Parallel()

	client := &Client{model: DefaultOllamaMultimodalModel}

	// Create test data
	testImageData := []byte("fake-jpeg-data")
	testFileData := []byte("file content")

	req := llm.ChatRequest{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []llm.MessageContent{
					llm.NewTextContent("Analyze this image and file:"),
					llm.NewImageContentFromBytes(testImageData, "image/jpeg"),
					llm.NewFileContentFromBytes(testFileData, "data.txt", "text/plain"),
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

func TestClient_ConvertRoleToOllama(t *testing.T) {
	t.Parallel()

	client := &Client{}

	tests := []struct {
		input    llm.MessageRole
		expected string
	}{
		{llm.RoleSystem, "system"},
		{llm.RoleUser, "user"},
		{llm.RoleAssistant, "assistant"},
		{llm.RoleTool, "assistant"}, // Ollama converts tool to assistant
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

func TestClient_ConvertRoleFromOllama(t *testing.T) {
	t.Parallel()

	client := &Client{}

	tests := []struct {
		input    string
		expected llm.MessageRole
	}{
		{"system", llm.RoleSystem},
		{"user", llm.RoleUser},
		{"assistant", llm.RoleAssistant},
		{"unknown", llm.RoleUser}, // Default fallback
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
