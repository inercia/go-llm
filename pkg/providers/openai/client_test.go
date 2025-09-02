package openai

import (
	"strings"
	"testing"

	"github.com/inercia/go-llm/pkg/llm"
)

// TestOpenAI_PublicAPI_BasicMultiModal tests the public API with multimodal content
func TestOpenAI_PublicAPI_BasicMultiModal(t *testing.T) {
	t.Parallel()

	// Test that we can create multimodal messages using the public API
	testImageData := []byte("test-image-data")

	message := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.MessageContent{
			llm.NewTextContent("Describe this image:"),
			llm.NewImageContentFromBytes(testImageData, "image/jpeg"),
		},
	}

	// Verify message structure
	if len(message.Content) != 2 {
		t.Errorf("Expected 2 content parts, got %d", len(message.Content))
	}

	// Verify text content exists
	textContent := message.Content[0]
	if textContent == nil {
		t.Error("First content should not be nil")
	}

	// Verify image content exists
	imageContent := message.Content[1]
	if imageContent == nil {
		t.Error("Second content should not be nil")
	}

	// Test message text extraction
	messageText := message.GetText()
	if !strings.Contains(messageText, "Describe this image:") {
		t.Error("Message text should contain the text content")
	}
}

// TestOpenAI_PublicAPI_FileContent tests file content handling
func TestOpenAI_PublicAPI_FileContent(t *testing.T) {
	t.Parallel()

	testFileData := []byte("Hello, world!")

	message := llm.Message{
		Role: llm.RoleUser,
		Content: []llm.MessageContent{
			llm.NewTextContent("Process this file:"),
			llm.NewFileContentFromBytes(testFileData, "test.txt", "text/plain"),
		},
	}

	// Verify message structure
	if len(message.Content) != 2 {
		t.Errorf("Expected 2 content parts, got %d", len(message.Content))
	}

	// Verify file content exists
	fileContent := message.Content[1]
	if fileContent == nil {
		t.Error("Second content should not be nil")
	}

	// Test that we can create a proper chat request
	req := llm.ChatRequest{
		Model:     "gpt-4o",
		Messages:  []llm.Message{message},
		MaxTokens: func() *int { i := 100; return &i }(),
	}

	// Verify request structure
	if len(req.Messages) != 1 {
		t.Errorf("Expected 1 message, got %d", len(req.Messages))
	}

	if req.Model != "gpt-4o" {
		t.Errorf("Expected model gpt-4o, got %s", req.Model)
	}
}

// TestOpenAI_PublicAPI_TextMessages tests simple text message creation
func TestOpenAI_PublicAPI_TextMessages(t *testing.T) {
	t.Parallel()

	// Test text message creation
	systemMsg := llm.NewTextMessage(llm.RoleSystem, "You are a helpful assistant")
	userMsg := llm.NewTextMessage(llm.RoleUser, "Hello")

	// Verify system message
	if systemMsg.Role != llm.RoleSystem {
		t.Errorf("Expected system role, got %s", systemMsg.Role)
	}

	systemText := systemMsg.GetText()
	if systemText != "You are a helpful assistant" {
		t.Errorf("Expected system text, got %s", systemText)
	}

	// Verify user message
	if userMsg.Role != llm.RoleUser {
		t.Errorf("Expected user role, got %s", userMsg.Role)
	}

	userText := userMsg.GetText()
	if userText != "Hello" {
		t.Errorf("Expected user text, got %s", userText)
	}

	// Test creating a complete chat request
	req := llm.ChatRequest{
		Model:       "gpt-4",
		Messages:    []llm.Message{systemMsg, userMsg},
		Temperature: func() *float32 { f := float32(0.7); return &f }(),
	}

	// Verify request
	if len(req.Messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(req.Messages))
	}

	if *req.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", *req.Temperature)
	}
}

// TestOpenAI_PublicAPI_ImageFromURL tests image content from URL
func TestOpenAI_PublicAPI_ImageFromURL(t *testing.T) {
	t.Parallel()

	imageURL := "https://example.com/test.jpg"
	imageContent := llm.NewImageContentFromURL(imageURL, "image/jpeg")

	message := llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.MessageContent{imageContent},
	}

	// Verify message structure
	if len(message.Content) != 1 {
		t.Errorf("Expected 1 content part, got %d", len(message.Content))
	}

	// Verify image content exists
	content := message.Content[0]
	if content == nil {
		t.Error("Content should not be nil")
	}

	// Verify we can create a chat request with image URL
	req := llm.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llm.Message{message},
	}

	if req.Model != "gpt-4o" {
		t.Errorf("Expected model gpt-4o, got %s", req.Model)
	}
}
