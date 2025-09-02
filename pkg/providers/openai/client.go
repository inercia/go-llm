package openai

import (
	"context"
	"io"
	"regexp"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/inercia/go-llm/pkg/llm"
)

// ModelAttribute represents a model attribute with its pattern and value
type ModelAttribute[T any] struct {
	Pattern *regexp.Regexp
	Value   T
}

// ModelAttributes contains all model attribute patterns
var (
	// Vision support patterns - models that support image inputs
	visionSupport = []ModelAttribute[bool]{
		{regexp.MustCompile(`^gpt-4o(-mini)?$`), true},                   // gpt-4o, gpt-4o-mini
		{regexp.MustCompile(`^gpt-4-turbo(-\d{4}-\d{2}-\d{2})?$`), true}, // gpt-4-turbo variants
		{regexp.MustCompile(`^gpt-4-vision-preview$`), true},             // gpt-4-vision-preview
		{regexp.MustCompile(`.*`), false},                                // Default: no vision support
	}

	// Tools support patterns - models that support function calling
	toolsSupport = []ModelAttribute[bool]{
		{regexp.MustCompile(`^gpt-4o(-mini)?$`), true},                            // gpt-4o, gpt-4o-mini
		{regexp.MustCompile(`^gpt-4(-0613|-32k|-32k-0613)?$`), true},              // gpt-4 variants
		{regexp.MustCompile(`^gpt-4-turbo(-preview|-\d{4}-\d{2}-\d{2})?$`), true}, // gpt-4-turbo variants
		{regexp.MustCompile(`^gpt-3\.5-turbo(-16k|-\d{4}-\d{2}-\d{2})?$`), true},  // gpt-3.5-turbo variants
		// For custom endpoints, check for GPT-like models
		{regexp.MustCompile(`(?i).*gpt.*`), true}, // Any GPT-like model
		{regexp.MustCompile(`(?i).*oss.*`), true}, // OSS models
		{regexp.MustCompile(`.*`), false},         // Default: no tools support
	}

	// Context length patterns - maximum tokens for different models
	contextLength = []ModelAttribute[int]{
		{regexp.MustCompile(`^gpt-4o(-mini)?$`), 128000},                            // gpt-4o series
		{regexp.MustCompile(`^gpt-4-turbo(-preview|-\d{4}-\d{2}-\d{2})?$`), 128000}, // gpt-4-turbo series
		{regexp.MustCompile(`^gpt-4-32k(-0613)?$`), 32768},                          // gpt-4-32k variants
		{regexp.MustCompile(`^gpt-4(-0613)?$`), 8192},                               // gpt-4 base variants
		{regexp.MustCompile(`^gpt-3\.5-turbo-16k(-\d{4}-\d{2}-\d{2})?$`), 16384},    // gpt-3.5-turbo-16k variants
		{regexp.MustCompile(`^gpt-3\.5-turbo(-\d{4}-\d{2}-\d{2})?$`), 4096},         // gpt-3.5-turbo base variants
		{regexp.MustCompile(`.*`), 4096},                                            // Default context length
	}
)

// getModelAttribute returns the attribute value for a given model by matching against patterns
func getModelAttribute[T any](model string, attributes []ModelAttribute[T]) T {
	for _, attr := range attributes {
		if attr.Pattern.MatchString(model) {
			return attr.Value
		}
	}
	// This should never be reached due to the catch-all pattern, but return zero value as fallback
	var zero T
	return zero
}

// Client implements the llm.Client interface for OpenAI
type Client struct {
	client   *openai.Client
	model    string
	provider string
	baseURL  string

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new OpenAI client
func NewClient(config llm.ClientConfig) (*Client, error) {
	if config.APIKey == "" {
		return nil, &llm.Error{
			Code:    "missing_api_key",
			Message: "API key is required for OpenAI",
			Type:    "authentication_error",
		}
	}

	clientConfig := openai.DefaultConfig(config.APIKey)
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}

	// Note: go-openai doesn't expose HTTPClient.Timeout directly
	// This would be handled differently in the actual implementation

	return &Client{
		client:   openai.NewClientWithConfig(clientConfig),
		model:    config.Model,
		provider: "openai",
		baseURL:  config.BaseURL,
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Auto-select appropriate model for multi-modal content
	model := c.selectModelForRequest(req)

	// Convert our request to OpenAI format
	openaiReq := c.convertRequest(req, model)

	// Make the actual API call
	resp, err := c.client.CreateChatCompletion(ctx, openaiReq)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Convert response back to our format
	return c.convertResponse(resp), nil
}

// StreamChatCompletion performs a streaming chat completion request using OpenAI
func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Auto-select appropriate model for multi-modal content
	model := c.selectModelForRequest(req)

	// Convert our request to OpenAI format (Stream is already set to true if requested)
	openaiReq := c.convertRequest(req, model)
	if !openaiReq.Stream {
		openaiReq.Stream = true
	}

	// Create the streaming request
	stream, err := c.client.CreateChatCompletionStream(ctx, openaiReq)
	if err != nil {
		return nil, c.convertError(err)
	}

	ch := make(chan llm.StreamEvent, 10)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		for {
			response, err := stream.Recv()
			if err == io.EOF {
				// Stream complete
				ch <- llm.NewDoneEvent(0, "stop")
				return
			}
			if err != nil {
				ch <- llm.NewErrorEvent(c.convertError(err))
				return
			}

			// Convert chunk to delta event
			delta := &llm.MessageDelta{}
			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				if choice.Delta.Content != "" {
					delta.Content = []llm.MessageContent{llm.NewTextContent(choice.Delta.Content)}
				}
				if choice.Delta.ToolCalls != nil {
					// Convert tool calls
					for i, tc := range choice.Delta.ToolCalls {
						toolCallDelta := llm.ToolCallDelta{
							Index: i,
							ID:    tc.ID,
							Type:  string(tc.Type),
						}
						if tc.Function.Name != "" {
							toolCallDelta.Function = &llm.ToolCallFunctionDelta{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							}
						}
						delta.ToolCalls = append(delta.ToolCalls, toolCallDelta)
					}
				}

				ch <- llm.NewDeltaEvent(0, delta)
			}
		}
	}()

	return ch, nil
}

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "openai",
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

// performHealthCheck performs a simple health check on the OpenAI API
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try to list models as a health check
	_, err := c.client.ListModels(ctx)
	return err == nil
}

// GetModelInfo returns information about the model being used
func (c *Client) GetModelInfo() llm.ModelInfo {
	return llm.ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         c.getMaxTokensForModel(c.model),
		SupportsTools:     c.supportsTools(c.model),
		SupportsVision:    c.supportsVision(c.model),
		SupportsFiles:     c.supportsFiles(c.model),
		SupportsStreaming: true,
	}
}

// Close cleans up any resources used by the client
func (c *Client) Close() error {
	// OpenAI client doesn't require explicit cleanup
	return nil
}

// selectModelForRequest automatically selects appropriate model based on content
func (c *Client) selectModelForRequest(req llm.ChatRequest) string {
	hasVision := false
	for _, msg := range req.Messages {
		if msg.HasContentType(llm.MessageTypeImage) {
			hasVision = true
			break
		}
	}

	if hasVision && !c.supportsVision(c.model) {
		// Auto-upgrade to vision model
		if c.model == "gpt-4" || c.model == "gpt-4-turbo" {
			return "gpt-4-vision-preview"
		}
		if c.model == "gpt-4o" || c.model == "gpt-4o-mini" {
			return c.model // These already support vision
		}
		return "gpt-4-vision-preview" // Safe default
	}

	return c.model
}

// convertRequest converts our ChatRequest to OpenAI format
func (c *Client) convertRequest(req llm.ChatRequest, model string) openai.ChatCompletionRequest {
	openaiReq := openai.ChatCompletionRequest{
		Model:    model,
		Messages: c.convertMessages(req.Messages),
		Stream:   req.Stream,
	}

	// Handle optional pointer fields
	if req.Temperature != nil {
		openaiReq.Temperature = *req.Temperature
	}
	if req.MaxTokens != nil {
		openaiReq.MaxTokens = *req.MaxTokens
	}
	if req.TopP != nil {
		openaiReq.TopP = *req.TopP
	}

	// Convert tools
	if len(req.Tools) > 0 {
		for _, tool := range req.Tools {
			openaiTool := openai.Tool{
				Type: openai.ToolType(tool.Type),
				Function: &openai.FunctionDefinition{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
			openaiReq.Tools = append(openaiReq.Tools, openaiTool)
		}
	}

	// Handle response format
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSON:
			openaiReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
				Type: openai.ChatCompletionResponseFormatTypeJSONObject,
			}
		case llm.ResponseFormatJSONSchema:
			if req.ResponseFormat.JSONSchema != nil {
				jsonSchema := &openai.ChatCompletionResponseFormatJSONSchema{
					Name:        req.ResponseFormat.JSONSchema.Name,
					Description: req.ResponseFormat.JSONSchema.Description,
				}
				// Handle Schema conversion - skip for now due to interface{} to json.Marshaler issue
				// TODO: Implement proper schema conversion
				// if req.ResponseFormat.JSONSchema.Schema != nil {
				//     jsonSchema.Schema = req.ResponseFormat.JSONSchema.Schema
				// }
				if req.ResponseFormat.JSONSchema.Strict != nil {
					jsonSchema.Strict = *req.ResponseFormat.JSONSchema.Strict
				}

				openaiReq.ResponseFormat = &openai.ChatCompletionResponseFormat{
					Type:       openai.ChatCompletionResponseFormatTypeJSONSchema,
					JSONSchema: jsonSchema,
				}
			}
		}
	}

	return openaiReq
}

// convertMessages converts our messages to OpenAI format
func (c *Client) convertMessages(messages []llm.Message) []openai.ChatCompletionMessage {
	var openaiMessages []openai.ChatCompletionMessage

	for _, msg := range messages {
		openaiMsg := openai.ChatCompletionMessage{
			Role: string(msg.Role),
		}

		// Handle tool calls
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				openaiTC := openai.ToolCall{
					ID:   tc.ID,
					Type: openai.ToolType(tc.Type),
					Function: openai.FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
				openaiMsg.ToolCalls = append(openaiMsg.ToolCalls, openaiTC)
			}
		}

		// Handle tool call ID for tool response messages
		if msg.ToolCallID != "" {
			openaiMsg.ToolCallID = msg.ToolCallID
		}

		// Handle content - always ensure Content field is set to avoid 'undefined'
		if len(msg.Content) == 1 && msg.IsTextOnly() {
			// Simple text message - validate that text is not empty
			text := msg.GetText()
			if strings.TrimSpace(text) == "" {
				// Use space for empty text to avoid 'undefined' error from API
				openaiMsg.Content = " "
			} else {
				openaiMsg.Content = text
			}
		} else if len(msg.Content) > 0 {
			// Multi-modal content
			var parts []openai.ChatMessagePart

			for _, content := range msg.Content {
				switch content.Type() {
				case llm.MessageTypeText:
					if textContent, ok := content.(*llm.TextContent); ok {
						text := textContent.GetText()
						// Skip empty text parts to avoid API errors
						if strings.TrimSpace(text) != "" {
							parts = append(parts, openai.ChatMessagePart{
								Type: openai.ChatMessagePartTypeText,
								Text: text,
							})
						}
					}
				case llm.MessageTypeImage:
					if imgContent, ok := content.(*llm.ImageContent); ok {
						imageURL := openai.ChatMessageImageURL{
							URL:    c.convertImageToDataURL(imgContent),
							Detail: openai.ImageURLDetailAuto,
						}
						parts = append(parts, openai.ChatMessagePart{
							Type:     openai.ChatMessagePartTypeImageURL,
							ImageURL: &imageURL,
						})
					}
				}
			}

			// For multi-modal content, use MultiContent only (don't set Content to avoid conflicts)
			// Handle edge case where all content parts were empty/filtered out
			if len(parts) == 0 {
				// All content was empty, use space to avoid undefined
				openaiMsg.Content = " "
			} else {
				openaiMsg.MultiContent = parts
				// Note: Don't set Content when using MultiContent - they can't be used simultaneously
			}
		} else {
			// No content - set a minimal content to avoid undefined
			// This covers: assistant messages with tool calls, tool response messages, etc.
			// Use a space instead of empty string to ensure it's not converted to undefined
			openaiMsg.Content = " "
		}

		openaiMessages = append(openaiMessages, openaiMsg)
	}

	return openaiMessages
}

// convertImageToDataURL converts image content to data URL
func (c *Client) convertImageToDataURL(imgContent *llm.ImageContent) string {
	// TODO: Implement proper image content handling
	// For now, return a placeholder to get basic functionality working
	return "data:image/jpeg;base64,placeholder"
}

// convertResponse converts OpenAI response to our format
func (c *Client) convertResponse(resp openai.ChatCompletionResponse) *llm.ChatResponse {
	chatResp := &llm.ChatResponse{
		ID:    resp.ID,
		Model: resp.Model,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}

	for _, choice := range resp.Choices {
		ourChoice := llm.Choice{
			Index:        choice.Index,
			Message:      c.convertMessage(choice.Message),
			FinishReason: string(choice.FinishReason),
		}
		chatResp.Choices = append(chatResp.Choices, ourChoice)
	}

	return chatResp
}

// convertMessage converts OpenAI message to our format
func (c *Client) convertMessage(msg openai.ChatCompletionMessage) llm.Message {
	ourMsg := llm.Message{
		Role: llm.MessageRole(msg.Role),
	}

	// Handle content - always initialize Content array
	if msg.Content != "" {
		ourMsg.Content = []llm.MessageContent{llm.NewTextContent(msg.Content)}
	} else {
		// Initialize empty content array to avoid nil issues
		ourMsg.Content = []llm.MessageContent{}
	}

	// Handle tool calls
	if len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			ourTC := llm.ToolCall{
				ID:   tc.ID,
				Type: string(tc.Type),
				Function: llm.ToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
			ourMsg.ToolCalls = append(ourMsg.ToolCalls, ourTC)
		}
	}

	// Handle tool call ID
	if msg.ToolCallID != "" {
		ourMsg.ToolCallID = msg.ToolCallID
	}

	return ourMsg
}

// convertError converts OpenAI error to our format
func (c *Client) convertError(err error) *llm.Error {
	// Try to parse as OpenAI API error
	if apiErr, ok := err.(*openai.APIError); ok {
		code := "unknown"
		if apiErr.Code != nil {
			if codeStr, ok := apiErr.Code.(string); ok {
				code = codeStr
			}
		}
		return &llm.Error{
			Code:       code,
			Message:    apiErr.Message,
			Type:       apiErr.Type,
			StatusCode: apiErr.HTTPStatusCode,
		}
	}

	// Generic error
	return &llm.Error{
		Code:    "unknown_error",
		Message: err.Error(),
		Type:    "api_error",
	}
}

// getMaxTokensForModel returns max tokens for the model
func (c *Client) getMaxTokensForModel(model string) int {
	return getModelAttribute(model, contextLength)
}

// supportsTools checks if model supports function calling
func (c *Client) supportsTools(model string) bool {
	// For custom endpoints, use the pattern-based approach with enhanced GPT/OSS detection
	if c.baseURL != "" && c.baseURL != "https://api.openai.com/v1" {
		return getModelAttribute(model, toolsSupport)
	}

	// For official OpenAI API, use pattern-based approach but exclude the generic GPT/OSS patterns
	for _, attr := range toolsSupport {
		// Skip the generic catch-all patterns for official API
		pattern := attr.Pattern.String()
		if strings.Contains(pattern, "(?i).*gpt.*") || strings.Contains(pattern, "(?i).*oss.*") {
			continue
		}
		if attr.Pattern.MatchString(model) {
			return attr.Value
		}
	}

	return false
}

// supportsVision checks if model supports vision inputs
func (c *Client) supportsVision(model string) bool {
	return getModelAttribute(model, visionSupport)
}

// supportsFiles checks if model supports file inputs
func (c *Client) supportsFiles(model string) bool {
	// Most OpenAI models support file content through context
	return true
}
