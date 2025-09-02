package llm

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestOpenAIClient_MultiModal_ModelSelection(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		clientModel   string
		request       ChatRequest
		expectedModel string
	}{
		{
			name:        "text_only_uses_configured_model",
			clientModel: "gpt-3.5-turbo",
			request: ChatRequest{
				Messages: []Message{
					NewTextMessage(RoleUser, "Hello world"),
				},
			},
			expectedModel: "gpt-3.5-turbo",
		},
		{
			name:        "text_only_with_request_model",
			clientModel: "gpt-3.5-turbo",
			request: ChatRequest{
				Model: "gpt-4",
				Messages: []Message{
					NewTextMessage(RoleUser, "Hello world"),
				},
			},
			expectedModel: "gpt-4",
		},
		{
			name:        "image_content_selects_vision_model",
			clientModel: "gpt-3.5-turbo",
			request: ChatRequest{
				Messages: []Message{
					{
						Role: RoleUser,
						Content: []MessageContent{
							NewTextContent("Describe this image"),
							NewImageContentFromURL("http://example.com/image.jpg", "image/jpeg"),
						},
					},
				},
			},
			expectedModel: "gpt-4o",
		},
		{
			name:        "vision_capable_client_model_preserved",
			clientModel: "gpt-4o-mini",
			request: ChatRequest{
				Messages: []Message{
					{
						Role: RoleUser,
						Content: []MessageContent{
							NewImageContentFromURL("http://example.com/image.jpg", "image/jpeg"),
						},
					},
				},
			},
			expectedModel: "gpt-4o-mini",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client := &OpenAIClient{model: tt.clientModel}
			model := client.selectModelForRequest(tt.request)
			if model != tt.expectedModel {
				t.Errorf("Expected model %s, got %s", tt.expectedModel, model)
			}
		})
	}
}

func TestOpenAIClient_ConvertMessage_TextOnly(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{}

	tests := []struct {
		name     string
		message  Message
		expected string
	}{
		{
			name:     "single_text_content",
			message:  NewTextMessage(RoleUser, "Hello world"),
			expected: "Hello world",
		},
		{
			name: "empty_content",
			message: Message{
				Role:    RoleUser,
				Content: []MessageContent{},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertMessage(tt.message)
			if result.Content != tt.expected {
				t.Errorf("Expected content %q, got %q", tt.expected, result.Content)
			}
			if len(result.MultiContent) != 0 {
				t.Errorf("Expected empty MultiContent for simple text, got %d items", len(result.MultiContent))
			}
		})
	}
}

func TestOpenAIClient_ConvertMessage_MultiModal(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{}

	// Create test image data
	testImageData := []byte("fake-image-data")

	tests := []struct {
		name                 string
		message              Message
		expectedPartCount    int
		expectedTextParts    int
		expectedImageParts   int
		expectedEmptyContent bool
	}{
		{
			name: "text_and_image",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Describe this image:"),
					NewImageContentFromBytes(testImageData, "image/jpeg"),
				},
			},
			expectedPartCount:    2,
			expectedTextParts:    1,
			expectedImageParts:   1,
			expectedEmptyContent: true,
		},
		{
			name: "multiple_text_parts",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("First part"),
					NewTextContent("Second part"),
				},
			},
			expectedPartCount:    2,
			expectedTextParts:    2,
			expectedImageParts:   0,
			expectedEmptyContent: true,
		},
		{
			name: "image_from_url",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewImageContentFromURL("https://example.com/test.png", "image/png"),
				},
			},
			expectedPartCount:    1,
			expectedTextParts:    0,
			expectedImageParts:   1,
			expectedEmptyContent: true,
		},
		{
			name: "file_content",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Process this file:"),
					NewFileContentFromBytes([]byte("sample text content"), "test.txt", "text/plain"),
				},
			},
			expectedPartCount:    2,
			expectedTextParts:    2, // File gets converted to text
			expectedImageParts:   0,
			expectedEmptyContent: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertMessage(tt.message)

			// Check basic structure
			if tt.expectedEmptyContent && result.Content != "" {
				t.Errorf("Expected empty Content field for multi-modal, got %q", result.Content)
			}

			if len(result.MultiContent) != tt.expectedPartCount {
				t.Errorf("Expected %d parts, got %d", tt.expectedPartCount, len(result.MultiContent))
			}

			// Count parts by type
			textParts := 0
			imageParts := 0

			for _, part := range result.MultiContent {
				switch part.Type {
				case "text":
					textParts++
				case "image_url":
					imageParts++
				}
			}

			if textParts != tt.expectedTextParts {
				t.Errorf("Expected %d text parts, got %d", tt.expectedTextParts, textParts)
			}
			if imageParts != tt.expectedImageParts {
				t.Errorf("Expected %d image parts, got %d", tt.expectedImageParts, imageParts)
			}
		})
	}
}

func TestOpenAIClient_ConvertImageContent(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{}

	tests := []struct {
		name         string
		imageContent *ImageContent
		expectError  bool
		expectedType string
		checkDataURL bool
	}{
		{
			name:         "image_from_url",
			imageContent: NewImageContentFromURL("https://example.com/test.png", "image/png"),
			expectError:  false,
			expectedType: "image_url",
		},
		{
			name:         "image_from_bytes",
			imageContent: NewImageContentFromBytes([]byte("test-data"), "image/jpeg"),
			expectError:  false,
			expectedType: "image_url",
			checkDataURL: true,
		},
		{
			name:         "nil_image",
			imageContent: nil,
			expectError:  true,
		},
		{
			name:         "empty_image",
			imageContent: &ImageContent{},
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := client.convertImageContent(tt.imageContent)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if string(result.Type) != tt.expectedType {
				t.Errorf("Expected type %s, got %s", tt.expectedType, result.Type)
			}

			if tt.checkDataURL {
				if !strings.HasPrefix(result.ImageURL.URL, "data:") {
					t.Error("Expected data URL for byte content")
				}
				if !strings.Contains(result.ImageURL.URL, "base64,") {
					t.Error("Expected base64 encoding in data URL")
				}
			}
		})
	}
}

func TestOpenAIClient_ConvertFileContent(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{}

	tests := []struct {
		name         string
		fileContent  *FileContent
		expectError  bool
		expectedType string
		checkText    string
	}{
		{
			name:         "text_file",
			fileContent:  NewFileContentFromBytes([]byte("Hello world"), "test.txt", "text/plain"),
			expectError:  false,
			expectedType: "text",
			checkText:    "Hello world",
		},
		{
			name:         "json_file",
			fileContent:  NewFileContentFromBytes([]byte(`{"key": "value"}`), "data.json", "application/json"),
			expectError:  false,
			expectedType: "text",
			checkText:    `{"key": "value"}`,
		},
		{
			name:         "pdf_file",
			fileContent:  NewFileContentFromBytes([]byte("pdf-binary-data"), "document.pdf", "application/pdf"),
			expectError:  false,
			expectedType: "text",
			checkText:    "[PDF File:",
		},
		{
			name:         "file_from_url",
			fileContent:  NewFileContentFromURL("https://example.com/file.txt", "file.txt", "text/plain", 1024),
			expectError:  false,
			expectedType: "text",
			checkText:    "[File Reference:",
		},
		{
			name:        "nil_file",
			fileContent: nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result, err := client.convertFileContent(tt.fileContent)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if string(result.Type) != tt.expectedType {
				t.Errorf("Expected type %s, got %s", tt.expectedType, result.Type)
			}

			if tt.checkText != "" && !strings.Contains(result.Text, tt.checkText) {
				t.Errorf("Expected text to contain %q, got %q", tt.checkText, result.Text)
			}
		})
	}
}

func TestOpenAIClient_IsVisionCapableModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model    string
		expected bool
	}{
		{"gpt-4o", true},
		{"gpt-4o-mini", true},
		{"gpt-4-vision-preview", true},
		{"gpt-4-turbo", true},
		{"gpt-4-turbo-preview", true},
		{"gpt-4", false},
		{"gpt-3.5-turbo", false},
		{"unknown-model", false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()

			result := isVisionCapableModel(tt.model)
			if result != tt.expected {
				t.Errorf("Expected %v for model %s, got %v", tt.expected, tt.model, result)
			}
		})
	}
}

func TestOpenAIClient_GetVisionCapableModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		clientModel   string
		expectedModel string
	}{
		{
			name:          "vision_capable_client_model",
			clientModel:   "gpt-4o-mini",
			expectedModel: "gpt-4o-mini",
		},
		{
			name:          "non_vision_client_model",
			clientModel:   "gpt-3.5-turbo",
			expectedModel: "gpt-4o",
		},
		{
			name:          "unknown_client_model",
			clientModel:   "unknown-model",
			expectedModel: "gpt-4o",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			client := &OpenAIClient{model: tt.clientModel}
			result := client.getVisionCapableModel()

			if result != tt.expectedModel {
				t.Errorf("Expected model %s, got %s", tt.expectedModel, result)
			}
		})
	}
}

func TestOpenAIClient_ConvertRequest_BackwardCompatibility(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{model: "gpt-4"}

	// Test backward compatibility with text-only messages
	req := ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			NewTextMessage(RoleSystem, "You are a helpful assistant"),
			NewTextMessage(RoleUser, "Hello"),
		},
		Temperature: func() *float32 { f := float32(0.7); return &f }(),
		MaxTokens:   func() *int { i := 1000; return &i }(),
	}

	result := client.convertRequest(req, "gpt-4")

	if result.Model != "gpt-4" {
		t.Errorf("Expected model gpt-4, got %s", result.Model)
	}

	if len(result.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(result.Messages))
	}

	// Verify first message (system)
	if result.Messages[0].Role != "system" {
		t.Errorf("Expected system role, got %s", result.Messages[0].Role)
	}
	if result.Messages[0].Content != "You are a helpful assistant" {
		t.Errorf("Expected system content, got %s", result.Messages[0].Content)
	}
	if len(result.Messages[0].MultiContent) != 0 {
		t.Errorf("Expected empty MultiContent for text-only, got %d", len(result.Messages[0].MultiContent))
	}

	// Verify second message (user)
	if result.Messages[1].Role != "user" {
		t.Errorf("Expected user role, got %s", result.Messages[1].Role)
	}
	if result.Messages[1].Content != "Hello" {
		t.Errorf("Expected user content, got %s", result.Messages[1].Content)
	}

	// Verify parameters
	if result.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", result.Temperature)
	}
	if result.MaxTokens != 1000 {
		t.Errorf("Expected max tokens 1000, got %d", result.MaxTokens)
	}
}

func TestOpenAIClient_ConvertRequest_MultiModal(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{model: "gpt-4o"}

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

	result := client.convertRequest(req, "gpt-4o")

	if len(result.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(result.Messages))
	}

	msg := result.Messages[0]
	if msg.Content != "" {
		t.Errorf("Expected empty Content field for multi-modal, got %q", msg.Content)
	}

	if len(msg.MultiContent) != 3 {
		t.Fatalf("Expected 3 content parts, got %d", len(msg.MultiContent))
	}

	// Verify text part
	textPart := msg.MultiContent[0]
	if string(textPart.Type) != "text" {
		t.Errorf("Expected first part to be text, got %s", textPart.Type)
	}
	if textPart.Text != "Analyze this image and file:" {
		t.Errorf("Expected text content, got %s", textPart.Text)
	}

	// Verify image part
	imagePart := msg.MultiContent[1]
	if string(imagePart.Type) != "image_url" {
		t.Errorf("Expected second part to be image_url, got %s", imagePart.Type)
	}
	if imagePart.ImageURL == nil {
		t.Error("Expected ImageURL to be set")
	} else {
		expectedDataURL := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(testImageData)
		if imagePart.ImageURL.URL != expectedDataURL {
			t.Errorf("Expected data URL, got %s", imagePart.ImageURL.URL)
		}
	}

	// Verify file part (converted to text)
	filePart := msg.MultiContent[2]
	if string(filePart.Type) != "text" {
		t.Errorf("Expected third part to be text, got %s", filePart.Type)
	}
	if !strings.Contains(filePart.Text, "file content") {
		t.Errorf("Expected file text content, got %s", filePart.Text)
	}
}

// Benchmark tests for performance
func BenchmarkOpenAIClient_ConvertMessage_TextOnly(b *testing.B) {
	client := &OpenAIClient{}
	msg := NewTextMessage(RoleUser, "Hello world")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertMessage(msg)
	}
}

func BenchmarkOpenAIClient_ConvertMessage_MultiModal(b *testing.B) {
	client := &OpenAIClient{}
	testData := []byte("test-image-data")

	msg := Message{
		Role: RoleUser,
		Content: []MessageContent{
			NewTextContent("Describe this:"),
			NewImageContentFromBytes(testData, "image/jpeg"),
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.convertMessage(msg)
	}
}

func BenchmarkOpenAIClient_SelectModelForRequest(b *testing.B) {
	client := &OpenAIClient{model: "gpt-4"}

	req := ChatRequest{
		Messages: []Message{
			{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Test"),
					NewImageContentFromURL("http://example.com/image.jpg", "image/jpeg"),
				},
			},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client.selectModelForRequest(req)
	}
}

// Edge case and error handling tests
func TestOpenAIClient_MultiModal_EdgeCases(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{model: "gpt-4o"}

	tests := []struct {
		name          string
		message       Message
		expectError   bool
		expectedParts int
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
			expectError:   false,
			expectedParts: 2,
		},
		{
			name: "multiple_images",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Compare these images:"),
					NewImageContentFromURL("https://example.com/image1.jpg", "image/jpeg"),
					NewImageContentFromURL("https://example.com/image2.png", "image/png"),
				},
			},
			expectError:   false,
			expectedParts: 3,
		},
		{
			name: "mixed_content_types",
			message: Message{
				Role: RoleUser,
				Content: []MessageContent{
					NewTextContent("Process this data:"),
					NewFileContentFromBytes([]byte(`{"data": "test"}`), "data.json", "application/json"),
					NewImageContentFromBytes([]byte("image-data"), "image/png"),
					NewTextContent("And provide a summary."),
				},
			},
			expectError:   false,
			expectedParts: 4, // All should convert to content parts
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := client.convertMessage(tt.message)

			if tt.expectError {
				// Would need actual error scenario for this
				return
			}

			if len(result.MultiContent) != tt.expectedParts {
				t.Errorf("Expected %d content parts, got %d", tt.expectedParts, len(result.MultiContent))
			}

			// Verify all parts have content
			for i, part := range result.MultiContent {
				switch part.Type {
				case "text":
					if part.Text == "" {
						t.Errorf("Part %d has empty text content", i)
					}
				case "image_url":
					if part.ImageURL == nil || part.ImageURL.URL == "" {
						t.Errorf("Part %d has empty image URL", i)
					}
				}
			}
		})
	}
}

func TestOpenAIClient_MultiModal_ErrorHandling(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{model: "gpt-4o"}

	tests := []struct {
		name         string
		imageContent *ImageContent
		expectError  bool
		checkError   string
	}{
		{
			name:         "nil_image_content",
			imageContent: nil,
			expectError:  true,
			checkError:   "nil",
		},
		{
			name: "image_without_data_or_url",
			imageContent: &ImageContent{
				MimeType: "image/jpeg",
			},
			expectError: true,
			checkError:  "neither URL nor data",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := client.convertImageContent(tt.imageContent)

			if tt.expectError {
				if err == nil {
					t.Error("Expected error, got nil")
				} else if tt.checkError != "" && !strings.Contains(err.Error(), tt.checkError) {
					t.Errorf("Expected error containing %q, got %q", tt.checkError, err.Error())
				}
			} else if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func TestOpenAIClient_MultiModal_DataURLGeneration(t *testing.T) {
	t.Parallel()

	client := &OpenAIClient{}
	testData := []byte("sample-image-data")
	imageContent := NewImageContentFromBytes(testData, "image/jpeg")

	result, err := client.convertImageContent(imageContent)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	expectedPrefix := "data:image/jpeg;base64,"
	if !strings.HasPrefix(result.ImageURL.URL, expectedPrefix) {
		t.Errorf("Expected URL to start with %q, got %q", expectedPrefix, result.ImageURL.URL)
	}

	// Verify base64 encoding
	base64Part := strings.TrimPrefix(result.ImageURL.URL, expectedPrefix)
	decoded, err := base64.StdEncoding.DecodeString(base64Part)
	if err != nil {
		t.Fatalf("Failed to decode base64: %v", err)
	}

	if string(decoded) != string(testData) {
		t.Error("Decoded data doesn't match original")
	}
}
