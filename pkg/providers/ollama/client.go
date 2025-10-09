package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/inercia/go-llm/pkg/llm"
)

const DefaultOllamaModel = "gpt-oss:20b"

const DefaultOllamaBaseURL = "http://localhost:11434"

// modelCapabilities defines the capabilities for a model pattern
type modelCapabilities struct {
	pattern        *regexp.Regexp
	maxTokens      int
	supportsTools  bool
	supportsVision bool
	supportsFiles  bool
}

// modelCapabilitiesList defines capabilities for different Ollama models
// Models are matched in order, first match wins
var modelCapabilitiesList = []modelCapabilities{
	// Llama 3.1 models (131K context)
	{
		pattern:        regexp.MustCompile(`llama3\.1`),
		maxTokens:      131072,
		supportsTools:  false,
		supportsVision: false,
		supportsFiles:  false,
	},
	// Qwen models (32K context)
	{
		pattern:        regexp.MustCompile(`qwen`),
		maxTokens:      32768,
		supportsTools:  false,
		supportsVision: false,
		supportsFiles:  false,
	},
	// CodeLlama models (16K context)
	{
		pattern:        regexp.MustCompile(`codellama`),
		maxTokens:      16384,
		supportsTools:  false,
		supportsVision: false,
		supportsFiles:  false,
	},
	// Vision models (LLaVA, etc.)
	{
		pattern:        regexp.MustCompile(`llava|vision`),
		maxTokens:      4096,
		supportsTools:  false,
		supportsVision: true,
		supportsFiles:  false,
	},
}

// Client implements the llm.Client interface for Ollama
type Client struct {
	model      string
	baseURL    string
	httpClient *http.Client

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new Ollama client
func NewClient(config llm.ClientConfig) (*Client, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = DefaultOllamaBaseURL
	}

	model := config.Model
	if model == "" {
		model = DefaultOllamaModel
	}

	// Ensure the base URL doesn't have trailing slash
	baseURL = strings.TrimSuffix(baseURL, "/")

	timeout := config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second // Ollama can be slower for local inference
	}

	return &Client{
		model:   model,
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

// ChatCompletion performs a chat completion request using Ollama's API
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Convert to Ollama format
	ollamaReq := c.convertToOllamaRequest(req)

	// Build URL - Ollama uses /api/chat endpoint
	url := fmt.Sprintf("%s/api/chat", c.baseURL)

	// Serialize request
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &llm.Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to serialize request: %v", err),
			Type:    "client_error",
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, &llm.Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to create request: %v", err),
			Type:    "client_error",
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.Error{
			Code:    "network_error",
			Message: fmt.Sprintf("Request failed: %v", err),
			Type:    "network_error",
		}
	}
	defer func() { _ = resp.Body.Close() }()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &llm.Error{
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
		return nil, &llm.Error{
			Code:    "parse_error",
			Message: fmt.Sprintf("Failed to parse response: %v", err),
			Type:    "client_error",
		}
	}

	// Convert to our format
	return c.convertFromOllamaResponse(ollamaResp), nil
}

// StreamChatCompletion performs a streaming chat completion request using Ollama
func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Convert to Ollama format with stream enabled
	ollamaReq := c.convertToOllamaRequest(req)
	ollamaReq.Stream = true

	// Build URL
	url := fmt.Sprintf("%s/api/chat", c.baseURL)

	// Serialize request
	reqBody, err := json.Marshal(ollamaReq)
	if err != nil {
		return nil, &llm.Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to serialize request: %v", err),
			Type:    "client_error",
		}
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, &llm.Error{
			Code:    "request_error",
			Message: fmt.Sprintf("Failed to create request: %v", err),
			Type:    "client_error",
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, &llm.Error{
			Code:    "network_error",
			Message: fmt.Sprintf("Request failed: %v", err),
			Type:    "network_error",
		}
	}

	ch := make(chan llm.StreamEvent, 10)

	go func() {
		defer close(ch)
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			ch <- llm.NewErrorEvent(c.convertOllamaError(body, resp.StatusCode))
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
				ch <- llm.NewErrorEvent(&llm.Error{
					Code:    "parse_error",
					Message: fmt.Sprintf("Failed to parse chunk: %v", err),
					Type:    "client_error",
				})
				return
			}

			if ollamaChunk.Done {
				ch <- llm.NewDoneEvent(0, "stop")
				return
			}

			if ollamaChunk.Message.Content != "" {
				// Append to accumulated content
				accumulatedContent += ollamaChunk.Message.Content
				delta := &llm.MessageDelta{
					Content: []llm.MessageContent{llm.NewTextContent(ollamaChunk.Message.Content)},
				}
				ch <- llm.NewDeltaEvent(0, delta)
			}

			// Ollama doesn't support streaming tool calls, so skip if present
		}

		if err := scanner.Err(); err != nil {
			ch <- llm.NewErrorEvent(&llm.Error{
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

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "ollama",
	}

	// Check if we need to refresh the health status
	now := time.Now()
	needsRefresh := c.lastHealthCheck == nil ||
		now.Sub(*c.lastHealthCheck) >= llm.DefaultHealthCheckInterval

	if needsRefresh {
		healthy := c.performHealthCheck()
		c.lastHealthStatus = &healthy
		c.lastHealthCheck = &now
	}

	info.Status = &llm.ClientRemoteInfoStatus{
		Healthy:     c.lastHealthStatus,
		LastChecked: c.lastHealthCheck,
	}

	return info
}

// performHealthCheck performs a simple health check on the Ollama API
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Build URL for health check
	url := fmt.Sprintf("%s/api/tags", c.baseURL)

	// Create HTTP request for listing models (lightweight check)
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false
	}

	// Make request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return false
	}
	defer func() { _ = resp.Body.Close() }()

	// Check if we got a successful response
	return resp.StatusCode == http.StatusOK
}

// GetModelInfo returns information about the model
func (c *Client) GetModelInfo() llm.ModelInfo {
	// Default capabilities
	caps := modelCapabilities{
		maxTokens:      4096,
		supportsTools:  false,
		supportsVision: false,
		supportsFiles:  false,
	}

	// Find matching model capabilities
	for _, modelCaps := range modelCapabilitiesList {
		if modelCaps.pattern.MatchString(c.model) {
			caps = modelCaps
			break
		}
	}

	return llm.ModelInfo{
		Name:              c.model,
		Provider:          "ollama",
		MaxTokens:         caps.maxTokens,
		SupportsTools:     caps.supportsTools,
		SupportsVision:    caps.supportsVision,
		SupportsFiles:     caps.supportsFiles,
		SupportsStreaming: true,
	}
}

// Close cleans up resources
func (c *Client) Close() error {
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
func (c *Client) convertToOllamaRequest(req llm.ChatRequest) OllamaRequest {
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
				case llm.MessageTypeText:
					if text, ok := cont.(*llm.TextContent); ok {
						contentBuilder.WriteString(text.GetText())
					}
				case llm.MessageTypeImage:
					if img, ok := cont.(*llm.ImageContent); ok {
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
				case llm.MessageTypeFile:
					if file, ok := cont.(*llm.FileContent); ok {
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

	// Handle ResponseFormat by adding instructions to the system message
	// Ollama doesn't support structured outputs natively, so we use prompt engineering
	if req.ResponseFormat != nil {
		ollamaReq.Messages = c.addResponseFormatInstructions(ollamaReq.Messages, req.ResponseFormat)
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
func (c *Client) convertFromOllamaResponse(resp OllamaResponse) *llm.ChatResponse {
	choice := llm.Choice{
		Index: 0,
		Message: llm.Message{
			Role:    c.convertRoleFromOllama(resp.Message.Role),
			Content: []llm.MessageContent{llm.NewTextContent(resp.Message.Content)},
			// Ollama doesn't support tool calls in responses
			ToolCalls: nil,
		},
	}

	if resp.Done {
		choice.FinishReason = "stop"
	} else {
		choice.FinishReason = "length"
	}

	return &llm.ChatResponse{
		ID:      fmt.Sprintf("ollama-%d", time.Now().UnixNano()),
		Model:   resp.Model,
		Choices: []llm.Choice{choice},
		// Ollama doesn't provide usage information
		Usage: llm.Usage{},
	}
}

// Convert Ollama error to our standardized format
func (c *Client) convertOllamaError(body []byte, statusCode int) *llm.Error {
	// Try to parse as Ollama error format
	var ollamaErr OllamaError
	if err := json.Unmarshal(body, &ollamaErr); err == nil && ollamaErr.Error != "" {
		return &llm.Error{
			Code:       fmt.Sprintf("ollama_%d", statusCode),
			Message:    ollamaErr.Error,
			Type:       "api_error",
			StatusCode: statusCode,
		}
	}

	// Fallback for unparseable errors
	return &llm.Error{
		Code:       "ollama_error",
		Message:    fmt.Sprintf("HTTP %d: %s", statusCode, string(body)),
		Type:       "api_error",
		StatusCode: statusCode,
	}
}

// Helper functions
func (c *Client) convertRoleToOllama(role llm.MessageRole) string {
	switch role {
	case llm.RoleUser:
		return "user"
	case llm.RoleAssistant:
		return "assistant"
	case llm.RoleSystem:
		return "system"
	case llm.RoleTool:
		return "assistant" // Ollama doesn't have a separate tool role
	default:
		return "user"
	}
}

func (c *Client) convertRoleFromOllama(role string) llm.MessageRole {
	switch role {
	case "user":
		return llm.RoleUser
	case "assistant":
		return llm.RoleAssistant
	case "system":
		return llm.RoleSystem
	default:
		return llm.RoleUser
	}
}

// addResponseFormatInstructions adds JSON formatting instructions to the messages when ResponseFormat is specified
// Since Ollama doesn't support structured outputs natively, we use prompt engineering
func (c *Client) addResponseFormatInstructions(messages []OllamaMessage, responseFormat *llm.ResponseFormat) []OllamaMessage {
	if responseFormat == nil {
		return messages
	}

	var instruction string
	switch responseFormat.Type {
	case llm.ResponseFormatJSON:
		instruction = "Please respond only with valid JSON. Do not include any text before or after the JSON object."
	case llm.ResponseFormatJSONSchema:
		if responseFormat.JSONSchema != nil && responseFormat.JSONSchema.Schema != nil {
			// Convert schema to string for instruction
			schemaBytes, err := json.Marshal(responseFormat.JSONSchema.Schema)
			if err == nil {
				instruction = fmt.Sprintf("Please respond only with valid JSON that conforms to this schema: %s. Do not include any text before or after the JSON object.", string(schemaBytes))
			} else {
				instruction = "Please respond only with valid JSON. Do not include any text before or after the JSON object."
			}
		} else {
			instruction = "Please respond only with valid JSON. Do not include any text before or after the JSON object."
		}
	default:
		return messages // No formatting needed for text responses
	}

	// Add the instruction as a system message at the beginning
	systemMessage := OllamaMessage{
		Role:    "system",
		Content: instruction,
	}

	// Prepend the system instruction
	return append([]OllamaMessage{systemMessage}, messages...)
}

// Model capabilities are now handled by the centralized model registry
