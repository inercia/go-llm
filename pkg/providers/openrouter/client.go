package openrouter

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/revrost/go-openrouter"

	"github.com/inercia/go-llm/pkg/llm"
)

// Client implements the llm.Client interface for OpenRouter
type Client struct {
	client   *openrouter.Client
	model    string
	provider string
	config   llm.ClientConfig

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new OpenRouter client
func NewClient(config llm.ClientConfig) (*Client, error) {
	if config.APIKey == "" {
		return nil, &llm.Error{
			Code:    "missing_api_key",
			Message: "API key is required for OpenRouter",
			Type:    "authentication_error",
		}
	}

	// Create OpenRouter client configuration
	clientConfig := openrouter.DefaultConfig(config.APIKey)

	// Set custom base URL if provided
	if config.BaseURL != "" {
		clientConfig.BaseURL = config.BaseURL
	}

	// Set additional OpenRouter-specific configurations from Extra field
	if config.Extra != nil {
		if siteURL, ok := config.Extra["site_url"]; ok {
			clientConfig.HttpReferer = siteURL
		}
		if appName, ok := config.Extra["app_name"]; ok {
			clientConfig.XTitle = appName
		}
	}

	// Create the OpenRouter client
	client := openrouter.NewClientWithConfig(*clientConfig)

	return &Client{
		client:   client,
		model:    config.Model,
		provider: "openrouter",
		config:   config,
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Convert our request to OpenRouter format
	openrouterReq, err := c.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Make the actual API call
	resp, err := c.client.CreateChatCompletion(ctx, openrouterReq)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Convert response back to our format
	return c.convertResponse(resp), nil
}

// StreamChatCompletion performs a streaming chat completion request
func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Convert our request to OpenRouter format
	openrouterReq, err := c.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Ensure streaming is enabled
	openrouterReq.Stream = true

	// Create the streaming request
	stream, err := c.client.CreateChatCompletionStream(ctx, openrouterReq)
	if err != nil {
		return nil, c.convertError(err)
	}

	ch := make(chan llm.StreamEvent, 10)

	go func() {
		defer close(ch)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					// Stream complete
					ch <- llm.NewDoneEvent(0, "stop")
					return
				}
				ch <- llm.NewErrorEvent(c.convertError(err))
				return
			}

			// Convert chunk to delta event
			if streamEvent := c.convertStreamResponse(response); streamEvent != nil {
				ch <- *streamEvent
			}
		}
	}()

	return ch, nil
}

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "openrouter",
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

// performHealthCheck performs a simple health check on the OpenRouter API
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try to list models as a health check
	_, err := c.client.ListModels(ctx)
	return err == nil
}

// GetModelInfo returns information about the model
func (c *Client) GetModelInfo() llm.ModelInfo {
	// Default capabilities for OpenRouter models vary by underlying model
	return llm.ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         128000, // Conservative default
		SupportsTools:     true,   // Most OpenRouter models support tools
		SupportsVision:    true,   // Many OpenRouter models support vision
		SupportsFiles:     true,   // OpenRouter supports file inputs
		SupportsStreaming: true,   // OpenRouter supports streaming
	}
}

// Close cleans up resources
func (c *Client) Close() error {
	// The go-openrouter client manages its own HTTP client internally
	// and doesn't expose a Close method, so we don't need explicit cleanup.
	// However, we can set the client to nil to help with garbage collection
	// and prevent any potential use after close.
	if c.client != nil {
		c.client = nil
	}
	return nil
}

// convertRequest converts our llm.ChatRequest to OpenRouter format
func (c *Client) convertRequest(req llm.ChatRequest) (openrouter.ChatCompletionRequest, error) {
	// Use the model from the request if provided, otherwise use the client's model
	model := req.Model
	if model == "" {
		model = c.model
	}

	openrouterReq := openrouter.ChatCompletionRequest{
		Model:    model,
		Messages: make([]openrouter.ChatCompletionMessage, 0, len(req.Messages)),
		Stream:   req.Stream,
	}

	// Set optional parameters
	if req.Temperature != nil {
		openrouterReq.Temperature = *req.Temperature
	}
	if req.MaxTokens != nil {
		openrouterReq.MaxTokens = *req.MaxTokens
	}
	if req.TopP != nil {
		openrouterReq.TopP = *req.TopP
	}

	// Convert messages
	for _, msg := range req.Messages {
		openrouterMsg, err := c.convertMessage(msg)
		if err != nil {
			return openrouterReq, fmt.Errorf("failed to convert message: %w", err)
		}
		openrouterReq.Messages = append(openrouterReq.Messages, openrouterMsg)
	}

	// Convert tools if present
	if len(req.Tools) > 0 {
		// Validate that the model supports tools before sending them
		if err := c.validateToolSupport(req.Tools); err != nil {
			return openrouterReq, err
		}

		openrouterReq.Tools = make([]openrouter.Tool, 0, len(req.Tools))
		for _, tool := range req.Tools {
			openrouterTool := openrouter.Tool{
				Type: openrouter.ToolType(tool.Type),
				Function: &openrouter.FunctionDefinition{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  tool.Function.Parameters,
				},
			}
			openrouterReq.Tools = append(openrouterReq.Tools, openrouterTool)
		}
	}

	return openrouterReq, nil
}

// convertMessage converts our Message to OpenRouter format
func (c *Client) convertMessage(msg llm.Message) (openrouter.ChatCompletionMessage, error) {
	openrouterMsg := openrouter.ChatCompletionMessage{
		Role: string(msg.Role),
	}

	// Handle tool call ID for tool messages
	if msg.ToolCallID != "" {
		openrouterMsg.ToolCallID = msg.ToolCallID
	}

	// Handle tool calls
	if len(msg.ToolCalls) > 0 {
		openrouterMsg.ToolCalls = make([]openrouter.ToolCall, 0, len(msg.ToolCalls))
		for _, tc := range msg.ToolCalls {
			openrouterTC := openrouter.ToolCall{
				ID:   tc.ID,
				Type: openrouter.ToolType(tc.Type),
				Function: openrouter.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
			openrouterMsg.ToolCalls = append(openrouterMsg.ToolCalls, openrouterTC)
		}
	}

	// Convert content
	if len(msg.Content) == 0 {
		// Empty content
		openrouterMsg.Content = openrouter.Content{Text: ""}
	} else if len(msg.Content) == 1 && msg.Content[0].Type() == llm.MessageTypeText {
		// Simple text message
		if textContent, ok := msg.Content[0].(*llm.TextContent); ok {
			openrouterMsg.Content = openrouter.Content{Text: textContent.GetText()}
		}
	} else {
		// Multi-modal content
		err := c.validateMultiModalContent(msg.Content)
		if err != nil {
			return openrouterMsg, err
		}

		// Convert to multi-part content
		parts, err := c.convertContentParts(msg.Content)
		if err != nil {
			return openrouterMsg, err
		}

		openrouterMsg.Content = openrouter.Content{Multi: parts}
	}

	return openrouterMsg, nil
}

// convertResponse converts OpenRouter response to our format
func (c *Client) convertResponse(resp openrouter.ChatCompletionResponse) *llm.ChatResponse {
	response := &llm.ChatResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Choices: make([]llm.Choice, 0, len(resp.Choices)),
	}

	// Convert usage information
	if resp.Usage != nil {
		response.Usage = llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	// Convert choices
	for _, choice := range resp.Choices {
		ourChoice := llm.Choice{
			Index:        choice.Index,
			FinishReason: string(choice.FinishReason),
			Message: llm.Message{
				Role:    llm.MessageRole(choice.Message.Role),
				Content: []llm.MessageContent{llm.NewTextContent(choice.Message.Content.Text)},
			},
		}

		// Convert tool calls if present
		if len(choice.Message.ToolCalls) > 0 {
			ourChoice.Message.ToolCalls = make([]llm.ToolCall, 0, len(choice.Message.ToolCalls))
			for _, tc := range choice.Message.ToolCalls {
				ourTC := llm.ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: llm.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
				ourChoice.Message.ToolCalls = append(ourChoice.Message.ToolCalls, ourTC)
			}
		}

		response.Choices = append(response.Choices, ourChoice)
	}

	return response
}

// convertStreamResponse converts OpenRouter stream response to our llm.StreamEvent
func (c *Client) convertStreamResponse(resp openrouter.ChatCompletionStreamResponse) *llm.StreamEvent {
	if len(resp.Choices) == 0 {
		return nil
	}

	choice := resp.Choices[0]

	// Handle completion
	if choice.FinishReason != "" {
		event := llm.NewDoneEvent(choice.Index, string(choice.FinishReason))
		return &event
	}

	// Handle delta
	delta := &llm.MessageDelta{}
	hasContent := false

	// Convert content delta
	if choice.Delta.Content != "" {
		delta.Content = []llm.MessageContent{llm.NewTextContent(choice.Delta.Content)}
		hasContent = true
	}

	// Convert tool call deltas
	if len(choice.Delta.ToolCalls) > 0 {
		delta.ToolCalls = make([]llm.ToolCallDelta, 0, len(choice.Delta.ToolCalls))
		for _, tc := range choice.Delta.ToolCalls {
			index := 0
			if tc.Index != nil {
				index = *tc.Index
			}

			toolCallDelta := llm.ToolCallDelta{
				Index: index,
				ID:    tc.ID,
				Type:  string(tc.Type),
			}

			// Handle function call delta
			toolCallDelta.Function = &llm.ToolCallFunctionDelta{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}

			delta.ToolCalls = append(delta.ToolCalls, toolCallDelta)
		}
		hasContent = true
	}

	// Only send delta if there's actual content
	if hasContent {
		event := llm.NewDeltaEvent(choice.Index, delta)
		return &event
	}

	return nil
}

// validateMultiModalContent validates multi-modal content based on model capabilities
func (c *Client) validateMultiModalContent(content []llm.MessageContent) error {
	capabilities := c.GetModelInfo() // Use model info instead of disabled registry

	for _, item := range content {
		switch item.Type() {
		case llm.MessageTypeImage:
			if !capabilities.SupportsVision {
				return &llm.Error{
					Code:    "unsupported_content_type",
					Message: fmt.Sprintf("Model %s does not support vision/image content", c.model),
					Type:    "validation_error",
				}
			}

			// Validate image content
			if err := item.Validate(); err != nil {
				return &llm.Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid image content: %v", err),
					Type:    "validation_error",
				}
			}

		case llm.MessageTypeFile:
			if !capabilities.SupportsFiles {
				return &llm.Error{
					Code:    "unsupported_content_type",
					Message: fmt.Sprintf("Model %s does not support file content", c.model),
					Type:    "validation_error",
				}
			}

			// Validate file content
			if err := item.Validate(); err != nil {
				return &llm.Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid file content: %v", err),
					Type:    "validation_error",
				}
			}

		case llm.MessageTypeText:
			// Text is always supported, just validate
			if err := item.Validate(); err != nil {
				return &llm.Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid text content: %v", err),
					Type:    "validation_error",
				}
			}

		default:
			return &llm.Error{
				Code:    "unsupported_content_type",
				Message: fmt.Sprintf("Content type %s is not supported", item.Type()),
				Type:    "validation_error",
			}
		}
	}

	return nil
}

// convertContentParts converts our MessageContent items to OpenRouter ChatMessagePart format
func (c *Client) convertContentParts(content []llm.MessageContent) ([]openrouter.ChatMessagePart, error) {
	parts := make([]openrouter.ChatMessagePart, 0, len(content))

	for _, item := range content {
		switch typedContent := item.(type) {
		case *llm.TextContent:
			parts = append(parts, openrouter.ChatMessagePart{
				Type: openrouter.ChatMessagePartTypeText,
				Text: typedContent.GetText(),
			})

		case *llm.ImageContent:
			part, err := c.convertImageContent(typedContent)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)

		case *llm.FileContent:
			part, err := c.convertFileContent(typedContent)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)

		default:
			return nil, &llm.Error{
				Code:    "unsupported_content_type",
				Message: fmt.Sprintf("Cannot convert content type %s", item.Type()),
				Type:    "validation_error",
			}
		}
	}

	return parts, nil
}

// convertImageContent converts ImageContent to OpenRouter ChatMessagePart
func (c *Client) convertImageContent(img *llm.ImageContent) (openrouter.ChatMessagePart, error) {
	part := openrouter.ChatMessagePart{
		Type: openrouter.ChatMessagePartTypeImageURL,
	}

	if img.HasURL() {
		// Use URL reference
		part.ImageURL = &openrouter.ChatMessageImageURL{
			URL: img.URL,
		}
	} else if img.HasData() {
		// Convert binary data to data URL
		dataURL, err := c.convertImageDataToURL(img)
		if err != nil {
			return part, err
		}
		part.ImageURL = &openrouter.ChatMessageImageURL{
			URL: dataURL,
		}
	} else {
		return part, &llm.Error{
			Code:    "invalid_content",
			Message: "Image content must have either URL or binary data",
			Type:    "validation_error",
		}
	}

	return part, nil
}

// convertFileContent converts FileContent to OpenRouter ChatMessagePart
func (c *Client) convertFileContent(file *llm.FileContent) (openrouter.ChatMessagePart, error) {
	part := openrouter.ChatMessagePart{
		Type: openrouter.ChatMessagePartTypeFile,
	}

	if file.HasData() {
		// Convert binary data to base64
		fileData, err := c.convertFileDataToBase64(file)
		if err != nil {
			return part, err
		}
		part.File = &openrouter.FileContent{
			Filename: file.Filename,
			FileData: fileData,
		}
	} else if file.HasURL() {
		// OpenRouter doesn't support file URLs directly, so we need to return an error
		return part, &llm.Error{
			Code:    "unsupported_feature",
			Message: "OpenRouter does not support file URLs, only binary file data",
			Type:    "validation_error",
		}
	} else {
		return part, &llm.Error{
			Code:    "invalid_content",
			Message: "File content must have binary data",
			Type:    "validation_error",
		}
	}

	return part, nil
}

// convertImageDataToURL converts binary image data to a data URL
func (c *Client) convertImageDataToURL(img *llm.ImageContent) (string, error) {
	if !img.HasData() {
		return "", &llm.Error{
			Code:    "invalid_content",
			Message: "Image has no binary data to convert",
			Type:    "validation_error",
		}
	}

	// Validate MIME type
	if !llm.IsValidImageMimeType(img.MimeType) {
		return "", &llm.Error{
			Code:    "unsupported_mime_type",
			Message: fmt.Sprintf("Unsupported image MIME type: %s", img.MimeType),
			Type:    "validation_error",
		}
	}

	// Convert to base64 data URL
	base64Data := fmt.Sprintf("data:%s;base64,%s",
		img.MimeType,
		c.encodeBase64(img.Data))

	return base64Data, nil
}

// convertFileDataToBase64 converts binary file data to base64 string
func (c *Client) convertFileDataToBase64(file *llm.FileContent) (string, error) {
	if !file.HasData() {
		return "", &llm.Error{
			Code:    "invalid_content",
			Message: "File has no binary data to convert",
			Type:    "validation_error",
		}
	}

	// Validate MIME type
	if !llm.IsValidFileMimeType(file.MimeType) {
		return "", &llm.Error{
			Code:    "unsupported_mime_type",
			Message: fmt.Sprintf("Unsupported file MIME type: %s", file.MimeType),
			Type:    "validation_error",
		}
	}

	return c.encodeBase64(file.Data), nil
}

// encodeBase64 encodes binary data to base64 string
func (c *Client) encodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// validateToolSupport validates that the model supports tools before sending tool definitions
func (c *Client) validateToolSupport(tools []llm.Tool) error {
	if len(tools) == 0 {
		return nil
	}

	capabilities := c.GetModelInfo() // Use model info instead of disabled registry
	if !capabilities.SupportsTools {
		return &llm.Error{
			Code:    "unsupported_feature",
			Message: fmt.Sprintf("Model %s does not support tools/function calling", c.model),
			Type:    "validation_error",
		}
	}

	// Validate individual tool definitions
	for i, tool := range tools {
		if err := c.validateToolDefinition(tool); err != nil {
			return &llm.Error{
				Code:    "invalid_tool_definition",
				Message: fmt.Sprintf("Tool %d validation failed: %v", i, err),
				Type:    "validation_error",
			}
		}
	}

	return nil
}

// validateToolDefinition validates a single tool definition
func (c *Client) validateToolDefinition(tool llm.Tool) error {
	// Validate tool type
	if tool.Type == "" {
		return fmt.Errorf("tool type is required")
	}
	if tool.Type != "function" {
		return fmt.Errorf("unsupported tool type: %s (only 'function' is supported)", tool.Type)
	}

	// Validate function definition
	if tool.Function.Name == "" {
		return fmt.Errorf("function name is required")
	}
	if tool.Function.Description == "" {
		return fmt.Errorf("function description is required")
	}

	// Validate function name format (should be valid identifier)
	if !isValidFunctionName(tool.Function.Name) {
		return fmt.Errorf("invalid function name format: %s", tool.Function.Name)
	}

	// Parameters can be nil for functions with no parameters
	if tool.Function.Parameters != nil {
		// Basic validation that parameters is a valid object structure
		if err := validateParametersSchema(tool.Function.Parameters); err != nil {
			return fmt.Errorf("invalid parameters schema: %v", err)
		}
	}

	return nil
}

// isValidFunctionName checks if a function name follows valid identifier rules
func isValidFunctionName(name string) bool {
	if name == "" {
		return false
	}

	// Function names should start with a letter or underscore
	// and contain only letters, numbers, and underscores
	for i, r := range name {
		if i == 0 {
			if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && r != '_' {
				return false
			}
		} else {
			if (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') && r != '_' {
				return false
			}
		}
	}
	return true
}

// validateParametersSchema performs basic validation on the parameters schema
func validateParametersSchema(params interface{}) error {
	// Convert to map to check basic structure
	paramMap, ok := params.(map[string]interface{})
	if !ok {
		return fmt.Errorf("parameters must be an object")
	}

	// Check for required "type" field
	typeField, hasType := paramMap["type"]
	if !hasType {
		return fmt.Errorf("parameters schema must have a 'type' field")
	}

	typeStr, ok := typeField.(string)
	if !ok {
		return fmt.Errorf("parameters 'type' field must be a string")
	}

	// For now, we only support "object" type parameters
	if typeStr != "object" {
		return fmt.Errorf("parameters type must be 'object', got: %s", typeStr)
	}

	// If properties exist, they should be an object
	if properties, hasProps := paramMap["properties"]; hasProps {
		if _, ok := properties.(map[string]interface{}); !ok {
			return fmt.Errorf("parameters 'properties' field must be an object")
		}
	}

	// If required exists, it should be an array
	if required, hasRequired := paramMap["required"]; hasRequired {
		if _, ok := required.([]interface{}); !ok {
			// Also accept []string
			if _, ok := required.([]string); !ok {
				return fmt.Errorf("parameters 'required' field must be an array")
			}
		}
	}

	return nil
}

// convertError converts OpenRouter errors to our standardized Error format
func (c *Client) convertError(err error) *llm.Error {
	return convertOpenRouterError(err)
}

// convertOpenRouterError converts OpenRouter errors to our standardized Error format
func convertOpenRouterError(err error) *llm.Error {
	if err == nil {
		return nil
	}

	// Handle OpenRouter APIError
	if apiErr, ok := err.(*openrouter.APIError); ok {
		return convertAPIError(apiErr)
	}

	// Handle OpenRouter RequestError
	if reqErr, ok := err.(*openrouter.RequestError); ok {
		return convertRequestError(reqErr)
	}

	// Handle wrapped errors by checking the underlying error
	if unwrapped := errors.Unwrap(err); unwrapped != nil {
		if converted := convertOpenRouterError(unwrapped); converted != nil {
			return converted
		}
	}

	// Handle common network and context errors
	if converted := convertCommonError(err); converted != nil {
		return converted
	}

	// Generic error fallback
	return &llm.Error{
		Code:    "openrouter_error",
		Message: err.Error(),
		Type:    "api_error",
	}
}

// convertAPIError converts OpenRouter APIError to our Error format
func convertAPIError(apiErr *openrouter.APIError) *llm.Error {
	// Determine error type and code based on HTTP status and error content
	errorType := "api_error"
	errorCode := "openrouter_api_error"

	// Map HTTP status codes to error types and codes
	switch apiErr.HTTPStatusCode {
	case 400:
		errorType = "validation_error"
		errorCode = "bad_request"
	case 401:
		errorType = "authentication_error"
		errorCode = "invalid_api_key"
	case 403:
		errorType = "authentication_error"
		errorCode = "insufficient_permissions"
	case 404:
		errorType = "model_error"
		errorCode = "model_not_found"
	case 429:
		errorType = "rate_limit_error"
		errorCode = "rate_limit_exceeded"
	case 500, 502, 503, 504:
		errorType = "api_error"
		errorCode = "server_error"
	default:
		if apiErr.HTTPStatusCode >= 400 && apiErr.HTTPStatusCode < 500 {
			errorType = "validation_error"
			errorCode = "client_error"
		} else if apiErr.HTTPStatusCode >= 500 {
			errorType = "api_error"
			errorCode = "server_error"
		}
	}

	// Use the API error code if available and it's a string
	if apiErr.Code != nil {
		if codeStr, ok := apiErr.Code.(string); ok && codeStr != "" {
			errorCode = codeStr
		}
	}

	// Refine error type and code based on message content
	message := apiErr.Message
	messageLower := strings.ToLower(message)

	// Authentication errors
	if strings.Contains(messageLower, "api key") || strings.Contains(messageLower, "unauthorized") {
		errorType = "authentication_error"
		if strings.Contains(messageLower, "invalid") || strings.Contains(messageLower, "incorrect") {
			errorCode = "invalid_api_key"
		} else if strings.Contains(messageLower, "missing") {
			errorCode = "missing_api_key"
		}
	}

	// Rate limiting errors
	if strings.Contains(messageLower, "rate limit") || strings.Contains(messageLower, "too many requests") {
		errorType = "rate_limit_error"
		errorCode = "rate_limit_exceeded"
	}

	// Model errors
	if strings.Contains(messageLower, "model") {
		errorType = "model_error"
		if strings.Contains(messageLower, "not found") || strings.Contains(messageLower, "does not exist") {
			errorCode = "model_not_found"
		} else if strings.Contains(messageLower, "not supported") || strings.Contains(messageLower, "unsupported") {
			errorCode = "model_not_supported"
		} else if strings.Contains(messageLower, "overloaded") || strings.Contains(messageLower, "unavailable") {
			errorCode = "model_overloaded"
		}
	}

	// Content filtering errors
	if strings.Contains(messageLower, "content policy") || strings.Contains(messageLower, "filtered") {
		errorType = "validation_error"
		errorCode = "content_filtered"
	}

	// Token limit errors
	if strings.Contains(messageLower, "token") && (strings.Contains(messageLower, "limit") || strings.Contains(messageLower, "maximum")) {
		errorType = "validation_error"
		errorCode = "token_limit_exceeded"
	}

	// Context length errors
	if strings.Contains(messageLower, "context") && strings.Contains(messageLower, "length") {
		errorType = "validation_error"
		errorCode = "context_length_exceeded"
	}

	return &llm.Error{
		Code:       errorCode,
		Message:    message,
		Type:       errorType,
		StatusCode: apiErr.HTTPStatusCode,
	}
}

// convertRequestError converts OpenRouter RequestError to our Error format
func convertRequestError(reqErr *openrouter.RequestError) *llm.Error {
	errorType := "network_error"
	errorCode := "request_error"

	// Map HTTP status codes to error types and codes
	switch reqErr.HTTPStatusCode {
	case 400:
		errorType = "validation_error"
		errorCode = "bad_request"
	case 401:
		errorType = "authentication_error"
		errorCode = "unauthorized"
	case 403:
		errorType = "authentication_error"
		errorCode = "forbidden"
	case 404:
		errorType = "model_error"
		errorCode = "not_found"
	case 429:
		errorType = "rate_limit_error"
		errorCode = "rate_limit_exceeded"
	case 500, 502, 503, 504:
		errorType = "api_error"
		errorCode = "server_error"
	default:
		if reqErr.HTTPStatusCode >= 400 && reqErr.HTTPStatusCode < 500 {
			errorType = "validation_error"
			errorCode = "client_error"
		} else if reqErr.HTTPStatusCode >= 500 {
			errorType = "api_error"
			errorCode = "server_error"
		}
	}

	// Use the full error message from RequestError.Error()
	// which includes status code, status, message, and body information
	message := reqErr.Error()

	return &llm.Error{
		Code:       errorCode,
		Message:    message,
		Type:       errorType,
		StatusCode: reqErr.HTTPStatusCode,
	}
}

// convertCommonError handles common Go errors that might occur during API calls
func convertCommonError(err error) *llm.Error {
	errMsg := err.Error()
	errMsgLower := strings.ToLower(errMsg)

	// Network connectivity errors
	if strings.Contains(errMsgLower, "connection refused") ||
		strings.Contains(errMsgLower, "no such host") ||
		strings.Contains(errMsgLower, "network is unreachable") {
		return &llm.Error{
			Code:    "connection_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// Timeout errors
	if strings.Contains(errMsgLower, "timeout") ||
		strings.Contains(errMsgLower, "deadline exceeded") {
		return &llm.Error{
			Code:    "timeout_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// Context cancellation
	if strings.Contains(errMsgLower, "context canceled") {
		return &llm.Error{
			Code:    "request_canceled",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// TLS/SSL errors
	if strings.Contains(errMsgLower, "tls") || strings.Contains(errMsgLower, "certificate") {
		return &llm.Error{
			Code:    "tls_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// DNS errors
	if strings.Contains(errMsgLower, "dns") {
		return &llm.Error{
			Code:    "dns_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	return nil
}

// OpenRouterModel represents a model from OpenRouter API
type OpenRouterModel struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Free        bool     `json:"free"`
	Inputs      []string `json:"inputs,omitempty"`
}

// ListModels retrieves available models from OpenRouter API
func (c *Client) ListModels(ctx context.Context) ([]OpenRouterModel, error) {
	resp, err := c.client.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	var models []OpenRouterModel
	for _, m := range resp { // Direct loop over slice
		// Determine if model is free based on pricing
		isFree := false
		if m.Pricing.Prompt == "0" && m.Pricing.Completion == "0" {
			isFree = true
		}

		// Create OpenRouterModel with available fields
		model := OpenRouterModel{
			ID:          m.ID,
			Name:        m.Name,
			Description: m.Description,
			Free:        isFree,
			Inputs:      m.Architecture.InputModalities,
		}

		models = append(models, model)
	}

	return models, nil
}

///////////////////////////////////////////////////////////////////////////////////////////////////

const OpenRouterFallbackTestingModel = "openai/gpt-3.5-turbo"

// OpenRouterModelPreferences is a list of regular expressions for matching models.
// The first model that matches will be returned.
// If no model matches, the fallback model will be returned.
var OpenRouterModelPreferences = []string{
	//"deepseek/deepseek.*r1.*",
	//"google/gemma-3n.*",
	"qwen/qwen3.*",
	//"google/gemini.*",
}

// GetOpenRouterTestingModel retrieves a testing model from OpenRouter API
// This is used for testing purposes only.
//
// Parameters:
// - free: if true, return a free model
// - vision: if true, return a model that supports vision
func GetOpenRouterTestingModel(free bool, vision bool) string {
	// Get test model from environment if set
	model := os.Getenv("OPENROUTER_TEST_MODEL")
	if model != "" {
		return model
	}

	// Check if we have an API key available
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		// No API key available, return fallback
		return OpenRouterFallbackTestingModel
	}

	// Create a temporary client to list models
	config := llm.ClientConfig{
		Provider: "openrouter",
		APIKey:   os.Getenv("OPENROUTER_API_KEY"),
		Model:    OpenRouterFallbackTestingModel, // fallback
	}

	client, err := NewClient(config)
	if err != nil {
		// If client creation fails, return fallback
		return OpenRouterFallbackTestingModel
	}
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	models, err := client.ListModels(ctx)
	if err != nil {
		// If listing fails, return fallback
		return OpenRouterFallbackTestingModel
	}

	// sort the list of model by name (for determinist behavior)
	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	if free {
		t := []OpenRouterModel{}
		// Find models that are free
		for _, model := range models {
			if !model.Free {
				// Skip models that are not free
				continue
			}
			t = append(t, model)
		}
		models = t
	}

	if vision {
		t := []OpenRouterModel{}
		// Find models that support vision
		for _, model := range models {
			supportsVision := false
			for _, input := range model.Inputs {
				if input == "image" {
					supportsVision = true
					break
				}
			}
			if supportsVision {
				t = append(t, model)
			}
		}
		models = t
	}

	// If models found after filtering, use ModelPreferences to select the best one
	if len(models) > 0 {
		// Try each preference regex in order
		for _, preference := range OpenRouterModelPreferences {
			for _, model := range models {
				if regexp.MustCompile(preference).MatchString(model.ID) {
					return model.ID
				}
			}
		}
		// If no preferences match, return the first one
		return models[0].ID
	}

	// Ultimate fallback
	return OpenRouterFallbackTestingModel
}
