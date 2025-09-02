package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaClient implements the Client interface for Ollama
type OllamaClient struct {
	model      string
	baseURL    string
	httpClient *http.Client
}

// NewOllamaClient creates a new Ollama client
func NewOllamaClient(config ClientConfig) (*OllamaClient, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	// Ensure the base URL doesn't have trailing slash
	baseURL = strings.TrimSuffix(baseURL, "/")

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second // Ollama can be slower for local inference
	}

	return &OllamaClient{
		model:   config.Model,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// ChatCompletion performs a chat completion request using Ollama's API
func (c *OllamaClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Convert to Ollama format
	ollamaReq := c.convertToOllamaRequest(req)

	// Build URL - Ollama uses /api/chat endpoint
	url := fmt.Sprintf("%s/api/chat", c.baseURL)

	// Serialize request
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to serialize request: %v", err),
			Type:    "client_error",
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, &Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to create request: %v", err),
			Type:    "client_error",
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, &Error{
			Code:    "network_error",
			Message: fmt.Sprintf("Request failed: %v", err),
			Type:    "network_error",
		}
	}
	defer func() { _ = resp.Body.Close() }()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &Error{
			Code:    "response_error",
			Message: fmt.Sprintf("Failed to read response: %v", err),
			Type:    "client_error",
		}
	}

	// Handle error responses
	if resp.StatusCode != http.StatusOK {
		return nil, c.convertOllamaError(body, resp.StatusCode)
	}

	// Parse successful response
	var ollamaResp OllamaResponse
	if err := json.Unmarshal(body, &ollamaResp); err != nil {
		return nil, &Error{
			Code:    "parse_error",
			Message: fmt.Sprintf("Failed to parse response: %v", err),
			Type:    "client_error",
		}
	}

	// Convert to our format
	return c.convertFromOllamaResponse(ollamaResp), nil
}

// StreamChatCompletion performs a streaming chat completion request using Ollama
func (c *OllamaClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
	// Convert to Ollama format with stream enabled
	ollamaReq := c.convertToOllamaRequest(req)
	ollamaReq.Stream = true

	// Build URL
	url := fmt.Sprintf("%s/api/chat", c.baseURL)

	// Serialize request
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to serialize request: %v", err),
			Type:    "client_error",
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, &Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to create request: %v", err),
			Type:    "client_error",
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, &Error{
			Code:    "network_error",
			Message: fmt.Sprintf("Request failed: %v", err),
			Type:    "network_error",
		}
	}

	ch := make(chan StreamEvent, 10)

	go func() {
		defer close(ch)
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			ch <- NewErrorEvent(c.convertOllamaError(body, resp.StatusCode))
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		var accumulatedContent string
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			// Ollama streams NDJSON, but lines are prefixed with "data: "
			line = strings.TrimPrefix(line, "data: ")

			var ollamaChunk OllamaStreamChunk
			if err := json.Unmarshal([]byte(line), &ollamaChunk); err != nil {
				ch <- NewErrorEvent(&Error{
					Code:    "parse_error",
					Message: fmt.Sprintf("Failed to parse chunk: %v", err),
					Type:    "client_error",
				})
				return
			}

			if ollamaChunk.Done {
				ch <- NewDoneEvent(0, "stop")
				return
			}

			if ollamaChunk.Message.Content != "" {
				// Append to accumulated content
				accumulatedContent += ollamaChunk.Message.Content
				delta := &MessageDelta{
					Content: []MessageContent{NewTextContent(ollamaChunk.Message.Content)},
				}
				ch <- NewDeltaEvent(0, delta)
			}

			// Ollama doesn't support streaming tool calls, so skip if present
		}

		if err := scanner.Err(); err != nil {
			ch <- NewErrorEvent(&Error{
				Code:    "stream_error",
				Message: fmt.Sprintf("Stream scan error: %v", err),
				Type:    "client_error",
			})
		}
	}()

	return ch, nil
}

// OllamaStreamChunk represents a streaming chunk from Ollama
type OllamaStreamChunk struct {
	Model   string        `json:"model"`
	Message OllamaMessage `json:"message"`
	Done    bool          `json:"done"`
	Error   string        `json:"error,omitempty"`
}

// GetModelInfo returns information about the model
func (c *OllamaClient) GetModelInfo() ModelInfo {
	capabilities := modelRegistry.GetModelCapabilities("ollama", c.model)
	return ModelInfo{
		Name:              c.model,
		Provider:          "ollama",
		MaxTokens:         capabilities.MaxTokens,
		SupportsTools:     capabilities.SupportsTools,
		SupportsVision:    capabilities.SupportsVision,
		SupportsFiles:     capabilities.SupportsFiles,
		SupportsStreaming: capabilities.SupportsStreaming,
	}
}

// Close cleans up resources
func (c *OllamaClient) Close() error {
	// No cleanup needed for HTTP client
	return nil
}

// Ollama API structures
type OllamaRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Options  *OllamaOptions  `json:"options,omitempty"`
	Images   []string        `json:"images,omitempty"` // Base64 encoded images for vision models
}

type OllamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaOptions struct {
	Temperature *float32 `json:"temperature,omitempty"`
	TopP        *float32 `json:"top_p,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"` // Ollama's equivalent to max_tokens
}

type OllamaResponse struct {
	Model   string        `json:"model"`
	Message OllamaMessage `json:"message"`
	Done    bool          `json:"done"`
	Error   string        `json:"error,omitempty"`
}

// Ollama error structure
type OllamaError struct {
	Error string `json:"error"`
}

// Convert our format to Ollama format
func (c *OllamaClient) convertToOllamaRequest(req ChatRequest) OllamaRequest {
	var images []string
	messages := make([]OllamaMessage, 0, len(req.Messages))

	for _, msg := range req.Messages {
		role := c.convertRoleToOllama(msg.Role)
		var contentBuilder strings.Builder

		// Handle multi-modal content
		if len(msg.Content) == 0 {
			contentBuilder.WriteString("")
		} else {
			for _, cont := range msg.Content {
				switch cont.Type() {
				case MessageTypeText:
					if text, ok := cont.(*TextContent); ok {
						contentBuilder.WriteString(text.GetText())
					}
				case MessageTypeImage:
					if img, ok := cont.(*ImageContent); ok {
						var imgData string
						if img.HasData() {
							imgData = base64.StdEncoding.EncodeToString(img.Data)
						} else if img.HasURL() {
							// For URLs, we'd need to fetch and encode, but for simplicity, skip or placeholder
							contentBuilder.WriteString(fmt.Sprintf("[Image: %s]", img.URL))
							continue
						}
						if imgData != "" {
							images = append(images, fmt.Sprintf("data:%s;base64,%s", img.MimeType, imgData))
							contentBuilder.WriteString("[Image attached]")
						}
					}
				case MessageTypeFile:
					if file, ok := cont.(*FileContent); ok {
						fileText := fmt.Sprintf("[File: %s, Type: %s, Size: %d bytes]", file.Filename, file.MimeType, file.FileSize)
						if file.HasURL() {
							fileText += fmt.Sprintf(" URL: %s", file.URL)
						}
						contentBuilder.WriteString(fileText)
					}
				}
			}
		}

		// Handle tool calls as text
		if len(msg.ToolCalls) > 0 {
			toolText := ""
			for _, toolCall := range msg.ToolCalls {
				toolText += fmt.Sprintf("\n[Tool Call: %s with args %s]",
					toolCall.Function.Name, toolCall.Function.Arguments)
			}
			contentBuilder.WriteString(toolText)
		}

		content := contentBuilder.String()
		if content != "" || role == "user" { // Ensure user messages are included even if empty
			messages = append(messages, OllamaMessage{
				Role:    role,
				Content: content,
			})
		}
	}

	ollamaReq := OllamaRequest{
		Model:    c.model, // Use client model if not specified
		Messages: messages,
		Stream:   req.Stream,
		Images:   images, // Add collected images at request level
	}

	// Add options if specified
	if req.Temperature != nil || req.MaxTokens != nil || req.TopP != nil {
		options := &OllamaOptions{}
		if req.Temperature != nil {
			temp := float32(*req.Temperature)
			options.Temperature = &temp
		}
		if req.MaxTokens != nil {
			options.NumPredict = req.MaxTokens
		}
		if req.TopP != nil {
			p := float32(*req.TopP)
			options.TopP = &p
		}
		ollamaReq.Options = options
	}

	return ollamaReq
}

// Convert Ollama response to our format
func (c *OllamaClient) convertFromOllamaResponse(resp OllamaResponse) *ChatResponse {
	choice := Choice{
		Index: 0,
		Message: Message{
			Role:    c.convertRoleFromOllama(resp.Message.Role),
			Content: []MessageContent{NewTextContent(resp.Message.Content)},
			// Ollama doesn't support tool calls in responses
			ToolCalls: nil,
		},
	}

	if resp.Done {
		choice.FinishReason = "stop"
	} else {
		choice.FinishReason = "length"
	}

	return &ChatResponse{
		ID:      fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Model:   resp.Model,
		Choices: []Choice{choice},
		// Ollama doesn't provide usage information
		Usage: Usage{},
	}
}

// Convert Ollama error to our standardized format
func (c *OllamaClient) convertOllamaError(body []byte, statusCode int) *Error {
	// Try to parse as Ollama error format
	var ollamaErr OllamaError
	if err := json.Unmarshal(body, &ollamaErr); err == nil && ollamaErr.Error != "" {
		return &Error{
			Code:       fmt.Sprintf("ollama_%d", statusCode),
			Message:    ollamaErr.Error,
			Type:       "api_error",
			StatusCode: statusCode,
		}
	}

	// Fallback for unparseable errors
	return &Error{
		Code:       "ollama_error",
		Message:    fmt.Sprintf("HTTP %d: %s", statusCode, string(body)),
		Type:       "api_error",
		StatusCode: statusCode,
	}
}

// Helper functions
func (c *OllamaClient) convertRoleToOllama(role MessageRole) string {
	switch role {
	case RoleUser:
		return "user"
	case RoleAssistant:
		return "assistant"
	case RoleSystem:
		return "system"
	case RoleTool:
		return "assistant" // Ollama doesn't have a separate tool role
	default:
		return "user"
	}
}

func (c *OllamaClient) convertRoleFromOllama(role string) MessageRole {
	switch role {
	case "user":
		return RoleUser
	case "assistant":
		return RoleAssistant
	case "system":
		return RoleSystem
	default:
		return RoleUser
	}
}

// Model capabilities are now handled by the centralized model registry
