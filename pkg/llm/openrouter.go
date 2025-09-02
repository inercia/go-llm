package llm

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/revrost/go-openrouter"
)

// OpenRouterClient implements the Client interface for OpenRouter
type OpenRouterClient struct {
	client   *openrouter.Client
	model    string
	provider string
	config   ClientConfig
}

// NewOpenRouterClient creates a new OpenRouter client
func NewOpenRouterClient(config ClientConfig) (*OpenRouterClient, error) {
	if config.APIKey == "" {
		return nil, &Error{
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

	return &OpenRouterClient{
		client:   client,
		model:    config.Model,
		provider: "openrouter",
		config:   config,
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *OpenRouterClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
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
func (c *OpenRouterClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
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

	ch := make(chan StreamEvent, 10)

	go func() {
		defer close(ch)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if err.Error() == "EOF" {
					// Stream complete
					ch <- NewDoneEvent(0, "stop")
					return
				}
				ch <- NewErrorEvent(c.convertError(err))
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

// GetModelInfo returns information about the model
func (c *OpenRouterClient) GetModelInfo() ModelInfo {
	capabilities := modelRegistry.GetModelCapabilities(c.provider, c.model)
	return ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         capabilities.MaxTokens,
		SupportsTools:     capabilities.SupportsTools,
		SupportsVision:    capabilities.SupportsVision,
		SupportsFiles:     capabilities.SupportsFiles,
		SupportsStreaming: capabilities.SupportsStreaming,
	}
}

// Close cleans up resources
func (c *OpenRouterClient) Close() error {
	// The go-openrouter client manages its own HTTP client internally
	// and doesn't expose a Close method, so we don't need explicit cleanup.
	// However, we can set the client to nil to help with garbage collection
	// and prevent any potential use after close.
	if c.client != nil {
		c.client = nil
	}
	return nil
}

// convertRequest converts our ChatRequest to OpenRouter format
func (c *OpenRouterClient) convertRequest(req ChatRequest) (openrouter.ChatCompletionRequest, error) {
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
func (c *OpenRouterClient) convertMessage(msg Message) (openrouter.ChatCompletionMessage, error) {
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
	} else if len(msg.Content) == 1 && msg.Content[0].Type() == MessageTypeText {
		// Simple text message
		if textContent, ok := msg.Content[0].(*TextContent); ok {
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
func (c *OpenRouterClient) convertResponse(resp openrouter.ChatCompletionResponse) *ChatResponse {
	response := &ChatResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Choices: make([]Choice, 0, len(resp.Choices)),
	}

	// Convert usage information
	if resp.Usage != nil {
		response.Usage = Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	// Convert choices
	for _, choice := range resp.Choices {
		ourChoice := Choice{
			Index:        choice.Index,
			FinishReason: string(choice.FinishReason),
			Message: Message{
				Role:    MessageRole(choice.Message.Role),
				Content: []MessageContent{NewTextContent(choice.Message.Content.Text)},
			},
		}

		// Convert tool calls if present
		if len(choice.Message.ToolCalls) > 0 {
			ourChoice.Message.ToolCalls = make([]ToolCall, 0, len(choice.Message.ToolCalls))
			for _, tc := range choice.Message.ToolCalls {
				ourTC := ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: ToolCallFunction{
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

// convertStreamResponse converts OpenRouter stream response to our StreamEvent
func (c *OpenRouterClient) convertStreamResponse(resp openrouter.ChatCompletionStreamResponse) *StreamEvent {
	if len(resp.Choices) == 0 {
		return nil
	}

	choice := resp.Choices[0]

	// Handle completion
	if choice.FinishReason != "" {
		return &StreamEvent{
			Type: "done",
			Choice: &StreamChoice{
				Index:        choice.Index,
				FinishReason: string(choice.FinishReason),
			},
		}
	}

	// Handle delta
	delta := &MessageDelta{}
	hasContent := false

	// Convert content delta
	if choice.Delta.Content != "" {
		delta.Content = []MessageContent{NewTextContent(choice.Delta.Content)}
		hasContent = true
	}

	// Convert tool call deltas
	if len(choice.Delta.ToolCalls) > 0 {
		delta.ToolCalls = make([]ToolCallDelta, 0, len(choice.Delta.ToolCalls))
		for _, tc := range choice.Delta.ToolCalls {
			index := 0
			if tc.Index != nil {
				index = *tc.Index
			}

			toolCallDelta := ToolCallDelta{
				Index: index,
				ID:    tc.ID,
				Type:  string(tc.Type),
			}

			// Handle function call delta
			toolCallDelta.Function = &ToolCallFunctionDelta{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}

			delta.ToolCalls = append(delta.ToolCalls, toolCallDelta)
		}
		hasContent = true
	}

	// Only send delta if there's actual content
	if hasContent {
		return &StreamEvent{
			Type: "delta",
			Choice: &StreamChoice{
				Index: choice.Index,
				Delta: delta,
			},
		}
	}

	return nil
}

// validateMultiModalContent validates multi-modal content based on model capabilities
func (c *OpenRouterClient) validateMultiModalContent(content []MessageContent) error {
	capabilities := modelRegistry.GetModelCapabilities(c.provider, c.model)

	for _, item := range content {
		switch item.Type() {
		case MessageTypeImage:
			if !capabilities.SupportsVision {
				return &Error{
					Code:    "unsupported_content_type",
					Message: fmt.Sprintf("Model %s does not support vision/image content", c.model),
					Type:    "validation_error",
				}
			}

			// Validate image content
			if err := item.Validate(); err != nil {
				return &Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid image content: %v", err),
					Type:    "validation_error",
				}
			}

		case MessageTypeFile:
			if !capabilities.SupportsFiles {
				return &Error{
					Code:    "unsupported_content_type",
					Message: fmt.Sprintf("Model %s does not support file content", c.model),
					Type:    "validation_error",
				}
			}

			// Validate file content
			if err := item.Validate(); err != nil {
				return &Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid file content: %v", err),
					Type:    "validation_error",
				}
			}

		case MessageTypeText:
			// Text is always supported, just validate
			if err := item.Validate(); err != nil {
				return &Error{
					Code:    "invalid_content",
					Message: fmt.Sprintf("Invalid text content: %v", err),
					Type:    "validation_error",
				}
			}

		default:
			return &Error{
				Code:    "unsupported_content_type",
				Message: fmt.Sprintf("Content type %s is not supported", item.Type()),
				Type:    "validation_error",
			}
		}
	}

	return nil
}

// convertContentParts converts our MessageContent items to OpenRouter ChatMessagePart format
func (c *OpenRouterClient) convertContentParts(content []MessageContent) ([]openrouter.ChatMessagePart, error) {
	parts := make([]openrouter.ChatMessagePart, 0, len(content))

	for _, item := range content {
		switch typedContent := item.(type) {
		case *TextContent:
			parts = append(parts, openrouter.ChatMessagePart{
				Type: openrouter.ChatMessagePartTypeText,
				Text: typedContent.GetText(),
			})

		case *ImageContent:
			part, err := c.convertImageContent(typedContent)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)

		case *FileContent:
			part, err := c.convertFileContent(typedContent)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)

		default:
			return nil, &Error{
				Code:    "unsupported_content_type",
				Message: fmt.Sprintf("Cannot convert content type %s", item.Type()),
				Type:    "validation_error",
			}
		}
	}

	return parts, nil
}

// convertImageContent converts ImageContent to OpenRouter ChatMessagePart
func (c *OpenRouterClient) convertImageContent(img *ImageContent) (openrouter.ChatMessagePart, error) {
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
		return part, &Error{
			Code:    "invalid_content",
			Message: "Image content must have either URL or binary data",
			Type:    "validation_error",
		}
	}

	return part, nil
}

// convertFileContent converts FileContent to OpenRouter ChatMessagePart
func (c *OpenRouterClient) convertFileContent(file *FileContent) (openrouter.ChatMessagePart, error) {
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
		return part, &Error{
			Code:    "unsupported_feature",
			Message: "OpenRouter does not support file URLs, only binary file data",
			Type:    "validation_error",
		}
	} else {
		return part, &Error{
			Code:    "invalid_content",
			Message: "File content must have binary data",
			Type:    "validation_error",
		}
	}

	return part, nil
}

// convertImageDataToURL converts binary image data to a data URL
func (c *OpenRouterClient) convertImageDataToURL(img *ImageContent) (string, error) {
	if !img.HasData() {
		return "", &Error{
			Code:    "invalid_content",
			Message: "Image has no binary data to convert",
			Type:    "validation_error",
		}
	}

	// Validate MIME type
	if !IsValidImageMimeType(img.MimeType) {
		return "", &Error{
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
func (c *OpenRouterClient) convertFileDataToBase64(file *FileContent) (string, error) {
	if !file.HasData() {
		return "", &Error{
			Code:    "invalid_content",
			Message: "File has no binary data to convert",
			Type:    "validation_error",
		}
	}

	// Validate MIME type
	if !IsValidFileMimeType(file.MimeType) {
		return "", &Error{
			Code:    "unsupported_mime_type",
			Message: fmt.Sprintf("Unsupported file MIME type: %s", file.MimeType),
			Type:    "validation_error",
		}
	}

	return c.encodeBase64(file.Data), nil
}

// encodeBase64 encodes binary data to base64 string
func (c *OpenRouterClient) encodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// validateToolSupport validates that the model supports tools before sending tool definitions
func (c *OpenRouterClient) validateToolSupport(tools []Tool) error {
	if len(tools) == 0 {
		return nil
	}

	capabilities := modelRegistry.GetModelCapabilities(c.provider, c.model)
	if !capabilities.SupportsTools {
		return &Error{
			Code:    "unsupported_feature",
			Message: fmt.Sprintf("Model %s does not support tools/function calling", c.model),
			Type:    "validation_error",
		}
	}

	// Validate individual tool definitions
	for i, tool := range tools {
		if err := c.validateToolDefinition(tool); err != nil {
			return &Error{
				Code:    "invalid_tool_definition",
				Message: fmt.Sprintf("Tool %d validation failed: %v", i, err),
				Type:    "validation_error",
			}
		}
	}

	return nil
}

// validateToolDefinition validates a single tool definition
func (c *OpenRouterClient) validateToolDefinition(tool Tool) error {
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
func (c *OpenRouterClient) convertError(err error) *Error {
	return convertOpenRouterError(err)
}

// convertOpenRouterError converts OpenRouter errors to our standardized Error format
func convertOpenRouterError(err error) *Error {
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
	return &Error{
		Code:    "openrouter_error",
		Message: err.Error(),
		Type:    "api_error",
	}
}

// convertAPIError converts OpenRouter APIError to our Error format
func convertAPIError(apiErr *openrouter.APIError) *Error {
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

	return &Error{
		Code:       errorCode,
		Message:    message,
		Type:       errorType,
		StatusCode: apiErr.HTTPStatusCode,
	}
}

// convertRequestError converts OpenRouter RequestError to our Error format
func convertRequestError(reqErr *openrouter.RequestError) *Error {
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

	return &Error{
		Code:       errorCode,
		Message:    message,
		Type:       errorType,
		StatusCode: reqErr.HTTPStatusCode,
	}
}

// convertCommonError handles common Go errors that might occur during API calls
func convertCommonError(err error) *Error {
	errMsg := err.Error()
	errMsgLower := strings.ToLower(errMsg)

	// Network connectivity errors
	if strings.Contains(errMsgLower, "connection refused") ||
		strings.Contains(errMsgLower, "no such host") ||
		strings.Contains(errMsgLower, "network is unreachable") {
		return &Error{
			Code:    "connection_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// Timeout errors
	if strings.Contains(errMsgLower, "timeout") ||
		strings.Contains(errMsgLower, "deadline exceeded") {
		return &Error{
			Code:    "timeout_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// Context cancellation
	if strings.Contains(errMsgLower, "context canceled") {
		return &Error{
			Code:    "request_canceled",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// TLS/SSL errors
	if strings.Contains(errMsgLower, "tls") || strings.Contains(errMsgLower, "certificate") {
		return &Error{
			Code:    "tls_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	// DNS errors
	if strings.Contains(errMsgLower, "dns") {
		return &Error{
			Code:    "dns_error",
			Message: errMsg,
			Type:    "network_error",
		}
	}

	return nil
}
