package deepseek

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/cohesion-org/deepseek-go"

	"github.com/inercia/go-llm/pkg/llm"
)

// Client implements the llm.Client interface for DeepSeek
type Client struct {
	client   *deepseek.Client
	model    string
	provider string
	config   llm.ClientConfig

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new DeepSeek client
func NewClient(config llm.ClientConfig) (*Client, error) {
	if config.APIKey == "" {
		return nil, &llm.Error{
			Code:    "missing_api_key",
			Message: "API key is required for DeepSeek",
			Type:    "authentication_error",
		}
	}

	// Validate model is provided
	if config.Model == "" {
		return nil, &llm.Error{
			Code:    "missing_model",
			Message: "model is required for DeepSeek client",
			Type:    "validation_error",
		}
	}

	// Prepare client options
	var opts []deepseek.Option

	// Set custom base URL if provided
	if config.BaseURL != "" {
		// Basic URL validation
		if config.BaseURL == "http://" || config.BaseURL == "https://" {
			return nil, &llm.Error{
				Code:    "invalid_base_url",
				Message: "base URL cannot be just a protocol",
				Type:    "validation_error",
			}
		}
		opts = append(opts, deepseek.WithBaseURL(config.BaseURL))
	}

	// Set timeout if provided
	if config.Timeout > 0 {
		opts = append(opts, deepseek.WithTimeout(config.Timeout))
	}

	// Create the DeepSeek client
	var client *deepseek.Client
	var err error

	if len(opts) > 0 {
		client, err = deepseek.NewClientWithOptions(config.APIKey, opts...)
		if err != nil {
			return nil, &llm.Error{
				Code:    "client_creation_error",
				Message: "Failed to create DeepSeek client: " + err.Error(),
				Type:    "configuration_error",
			}
		}
	} else {
		client = deepseek.NewClient(config.APIKey)
	}

	return &Client{
		client:   client,
		model:    config.Model,
		provider: "deepseek",
		config:   config,
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Convert our request to DeepSeek format
	deepseekReq, err := c.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Make the actual API call
	resp, err := c.client.CreateChatCompletion(ctx, &deepseekReq)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Convert response back to our format
	return c.convertResponse(*resp), nil
}

// StreamChatCompletion performs a streaming chat completion request
func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Convert our request to DeepSeek streaming format
	deepseekReq, err := c.convertStreamRequest(req)
	if err != nil {
		return nil, err
	}

	// Create the streaming request
	stream, err := c.client.CreateChatCompletionStream(ctx, &deepseekReq)
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

			// Convert chunk to stream event
			event := c.convertStreamEvent(response)
			if event != nil {
				ch <- *event
			}
		}
	}()

	return ch, nil
}

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "deepseek",
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

// performHealthCheck performs a simple health check on the DeepSeek API
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Simple test request with minimal parameters
	req := deepseek.ChatCompletionRequest{
		Model: c.model,
		Messages: []deepseek.ChatCompletionMessage{
			{
				Role:    "user",
				Content: "test",
			},
		},
		MaxTokens: 1,
	}

	_, err := c.client.CreateChatCompletion(ctx, &req)
	return err == nil
}

// GetModelInfo returns information about the model
func (c *Client) GetModelInfo() llm.ModelInfo {
	// Default capabilities for DeepSeek models
	return llm.ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         32768, // DeepSeek models typically support 32K context
		SupportsTools:     true,  // DeepSeek supports function calling
		SupportsVision:    false, // Most DeepSeek models are text-only
		SupportsFiles:     false, // Basic file support
		SupportsStreaming: true,  // DeepSeek supports streaming
	}
}

// Close cleans up resources
func (c *Client) Close() error {
	// The deepseek-go client manages its own HTTP client internally.
	// While it doesn't expose a Close method, we ensure that any
	// resources are properly handled by setting the client to nil
	// to help with garbage collection.
	if c.client != nil {
		c.client = nil
	}
	return nil
}

// convertRequest converts our llm.ChatRequest to DeepSeek format
func (c *Client) convertRequest(req llm.ChatRequest) (deepseek.ChatCompletionRequest, error) {
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		// Validate message security before processing (disabled for now)
		// if err := ValidateMessageSecurity(&msg); err != nil {
		//	return deepseek.ChatCompletionRequest{}, &llm.Error{
		//		Code:    "security_validation_failed",
		//		Message: "Message failed security validation: " + err.Error(),
		//		Type:    "validation_error",
		//	}
		// }
		convertedMsg, err := c.convertMessage(msg)
		if err != nil {
			return deepseek.ChatCompletionRequest{}, err
		}
		messages[i] = convertedMsg
	}

	// Convert tools if present
	var tools []deepseek.Tool
	if len(req.Tools) > 0 {
		// Check if model supports tools
		modelInfo := c.GetModelInfo()
		if !modelInfo.SupportsTools {
			return deepseek.ChatCompletionRequest{}, &llm.Error{
				Code:    "tools_not_supported",
				Message: "Model " + c.model + " does not support tools",
				Type:    "validation_error",
			}
		}

		tools = make([]deepseek.Tool, len(req.Tools))
		for i, tool := range req.Tools {
			// Convert parameters to FunctionParameters
			var params *deepseek.FunctionParameters
			if tool.Function.Parameters != nil {
				params = c.convertToolParameters(tool.Function.Parameters)
			}

			tools[i] = deepseek.Tool{
				Type: tool.Type,
				Function: deepseek.Function{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  params,
				},
			}
		}
	}

	deepseekReq := deepseek.ChatCompletionRequest{
		Model:    c.model,
		Messages: messages,
		Tools:    tools,
	}

	// Set optional parameters
	if req.Temperature != nil {
		deepseekReq.Temperature = *req.Temperature
	}
	if req.MaxTokens != nil {
		deepseekReq.MaxTokens = *req.MaxTokens
	}
	if req.TopP != nil {
		deepseekReq.TopP = *req.TopP
	}

	return deepseekReq, nil
}

// convertStreamRequest converts our llm.ChatRequest to DeepSeek streaming format
func (c *Client) convertStreamRequest(req llm.ChatRequest) (deepseek.StreamChatCompletionRequest, error) {
	messages := make([]deepseek.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		// Validate message security before processing (disabled for now)
		// if err := ValidateMessageSecurity(&msg); err != nil {
		//	return deepseek.StreamChatCompletionRequest{}, &llm.Error{
		//		Code:    "security_validation_failed",
		//		Message: "Message failed security validation: " + err.Error(),
		//		Type:    "validation_error",
		//	}
		// }
		convertedMsg, err := c.convertMessage(msg)
		if err != nil {
			return deepseek.StreamChatCompletionRequest{}, err
		}
		messages[i] = convertedMsg
	}

	// Convert tools if present
	var tools []deepseek.Tool
	if len(req.Tools) > 0 {
		// Check if model supports tools
		modelInfo := c.GetModelInfo()
		if !modelInfo.SupportsTools {
			return deepseek.StreamChatCompletionRequest{}, &llm.Error{
				Code:    "tools_not_supported",
				Message: "Model " + c.model + " does not support tools",
				Type:    "validation_error",
			}
		}

		tools = make([]deepseek.Tool, len(req.Tools))
		for i, tool := range req.Tools {
			// Convert parameters to FunctionParameters
			var params *deepseek.FunctionParameters
			if tool.Function.Parameters != nil {
				params = c.convertToolParameters(tool.Function.Parameters)
			}

			tools[i] = deepseek.Tool{
				Type: tool.Type,
				Function: deepseek.Function{
					Name:        tool.Function.Name,
					Description: tool.Function.Description,
					Parameters:  params,
				},
			}
		}
	}

	deepseekReq := deepseek.StreamChatCompletionRequest{
		Model:    c.model,
		Messages: messages,
		Tools:    tools,
		Stream:   true, // Always true for streaming requests
	}

	// Set optional parameters
	if req.Temperature != nil {
		deepseekReq.Temperature = *req.Temperature
	}
	if req.MaxTokens != nil {
		deepseekReq.MaxTokens = *req.MaxTokens
	}
	if req.TopP != nil {
		deepseekReq.TopP = *req.TopP
	}

	return deepseekReq, nil
}

// convertMessage converts our Message to DeepSeek format
func (c *Client) convertMessage(msg llm.Message) (deepseek.ChatCompletionMessage, error) {
	deepseekMsg := deepseek.ChatCompletionMessage{
		Role:       c.convertRoleToDeepSeek(msg.Role),
		ToolCalls:  c.convertToolCallsToDeepSeek(msg.ToolCalls),
		ToolCallID: msg.ToolCallID,
	}

	// Handle content - DeepSeek uses simple string content
	if len(msg.Content) == 0 {
		deepseekMsg.Content = ""
	} else {
		// Get model capabilities for validation
		modelInfo := c.GetModelInfo()

		// Validate total message size
		totalSize := msg.TotalSize()
		if err := c.validateMessageSize(totalSize); err != nil {
			return deepseek.ChatCompletionMessage{}, err
		}

		// Combine all content into a single string
		var contentBuilder []string

		for _, content := range msg.Content {
			switch content.Type() {
			case llm.MessageTypeText:
				if textContent, ok := content.(*llm.TextContent); ok {
					contentBuilder = append(contentBuilder, textContent.GetText())
				}
			case llm.MessageTypeImage:
				// Validate image content and model capabilities
				if err := c.validateImageContent(content.(*llm.ImageContent), modelInfo); err != nil {
					return deepseek.ChatCompletionMessage{}, err
				}

				convertedContent, err := c.convertImageContent(content.(*llm.ImageContent), modelInfo)
				if err != nil {
					return deepseek.ChatCompletionMessage{}, err
				}
				contentBuilder = append(contentBuilder, convertedContent)

			case llm.MessageTypeFile:
				// Validate file content and model capabilities
				if err := c.validateFileContent(content.(*llm.FileContent), modelInfo); err != nil {
					return deepseek.ChatCompletionMessage{}, err
				}

				convertedContent, err := c.convertFileContent(content.(*llm.FileContent), modelInfo)
				if err != nil {
					return deepseek.ChatCompletionMessage{}, err
				}
				contentBuilder = append(contentBuilder, convertedContent)
			}
		}

		// Join all content parts
		if len(contentBuilder) == 1 {
			deepseekMsg.Content = contentBuilder[0]
		} else if len(contentBuilder) > 1 {
			deepseekMsg.Content = ""
			for i, part := range contentBuilder {
				if i > 0 {
					deepseekMsg.Content += "\n\n"
				}
				deepseekMsg.Content += part
			}
		}
	}

	return deepseekMsg, nil
}

// convertRoleToDeepSeek converts our llm.MessageRole to DeepSeek format
func (c *Client) convertRoleToDeepSeek(role llm.MessageRole) string {
	switch role {
	case llm.RoleSystem:
		return "system"
	case llm.RoleUser:
		return "user"
	case llm.RoleAssistant:
		return "assistant"
	case llm.RoleTool:
		return "tool"
	default:
		return "user" // Default fallback
	}
}

// convertToolCallsToDeepSeek converts our ToolCalls to DeepSeek format
func (c *Client) convertToolCallsToDeepSeek(toolCalls []llm.ToolCall) []deepseek.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	deepseekToolCalls := make([]deepseek.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		deepseekToolCalls[i] = deepseek.ToolCall{
			Index: i, // DeepSeek requires an index
			ID:    tc.ID,
			Type:  tc.Type,
			Function: deepseek.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return deepseekToolCalls
}

// convertResponse converts DeepSeek response to our format
func (c *Client) convertResponse(resp deepseek.ChatCompletionResponse) *llm.ChatResponse {
	choices := make([]llm.Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		choices[i] = llm.Choice{
			Index: choice.Index,
			Message: llm.Message{
				Role:      c.convertRoleFromDeepSeek(choice.Message.Role),
				Content:   []llm.MessageContent{llm.NewTextContent(choice.Message.Content)},
				ToolCalls: c.convertToolCallsFromDeepSeek(choice.Message.ToolCalls),
			},
			FinishReason: choice.FinishReason,
		}
	}

	return &llm.ChatResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Choices: choices,
		Usage: llm.Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

// convertRoleFromDeepSeek converts DeepSeek role to our llm.MessageRole
func (c *Client) convertRoleFromDeepSeek(role string) llm.MessageRole {
	switch role {
	case "system":
		return llm.RoleSystem
	case "user":
		return llm.RoleUser
	case "assistant":
		return llm.RoleAssistant
	case "tool":
		return llm.RoleTool
	default:
		return llm.RoleAssistant // Default fallback
	}
}

// convertToolCallsFromDeepSeek converts DeepSeek ToolCalls to our format
func (c *Client) convertToolCallsFromDeepSeek(toolCalls []deepseek.ToolCall) []llm.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	ourToolCalls := make([]llm.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		ourToolCalls[i] = llm.ToolCall{
			ID:   tc.ID,
			Type: tc.Type,
			Function: llm.ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return ourToolCalls
}

// convertStreamEvent converts DeepSeek streaming response to llm.StreamEvent
func (c *Client) convertStreamEvent(resp *deepseek.StreamChatCompletionResponse) *llm.StreamEvent {
	if resp == nil || len(resp.Choices) == 0 {
		return nil
	}

	choice := resp.Choices[0]

	// Handle finish reason - if present, this is a done event
	if choice.FinishReason != "" {
		event := llm.NewDoneEvent(choice.Index, choice.FinishReason)
		return &event
	}

	// Handle delta content
	delta := &llm.MessageDelta{}
	hasContent := false

	// Handle text content delta
	if choice.Delta.Content != "" {
		delta.Content = []llm.MessageContent{llm.NewTextContent(choice.Delta.Content)}
		hasContent = true
	}

	// Handle tool calls delta
	if len(choice.Delta.ToolCalls) > 0 {
		for _, tc := range choice.Delta.ToolCalls {
			toolCallDelta := llm.ToolCallDelta{
				Index: tc.Index,
				ID:    tc.ID,
				Type:  tc.Type,
				Function: &llm.ToolCallFunctionDelta{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}

			delta.ToolCalls = append(delta.ToolCalls, toolCallDelta)
		}
		hasContent = true
	}

	// Only return delta event if there's actual content
	if hasContent {
		event := llm.NewDeltaEvent(choice.Index, delta)
		return &event
	}

	return nil
}

// convertError converts DeepSeek error to our standardized error format
func (c *Client) convertError(err error) *llm.Error {
	if err == nil {
		return nil
	}

	// Try to extract structured error information
	// This is a generic implementation - adjust based on actual deepseek-go error types
	errorMsg := err.Error()

	// Default error mapping
	code := "api_error"
	errorType := "api_error"
	statusCode := 0

	// Basic error classification based on error message
	switch {
	case contains(errorMsg, "unauthorized") || contains(errorMsg, "invalid api key") || contains(errorMsg, "authentication"):
		code = "authentication_error"
		errorType = "authentication_error"
		statusCode = 401
	case contains(errorMsg, "rate limit") || contains(errorMsg, "too many requests"):
		code = "rate_limit_error"
		errorType = "rate_limit_error"
		statusCode = 429
	case contains(errorMsg, "model") && contains(errorMsg, "not found"):
		code = "model_not_found"
		errorType = "model_error"
		statusCode = 404
	case contains(errorMsg, "timeout") || contains(errorMsg, "deadline"):
		code = "timeout_error"
		errorType = "network_error"
		statusCode = 408
	case contains(errorMsg, "validation") || contains(errorMsg, "invalid"):
		code = "validation_error"
		errorType = "validation_error"
		statusCode = 400
	}

	return &llm.Error{
		Code:       code,
		Message:    errorMsg,
		Type:       errorType,
		StatusCode: statusCode,
	}
}

// convertToolParameters converts interface{} parameters to DeepSeek FunctionParameters
func (c *Client) convertToolParameters(params interface{}) *deepseek.FunctionParameters {
	if params == nil {
		return nil
	}

	// Try to convert to map[string]interface{} first
	paramMap, ok := params.(map[string]interface{})
	if !ok {
		// If it's not a map, return a basic object type
		return &deepseek.FunctionParameters{
			Type: "object",
		}
	}

	result := &deepseek.FunctionParameters{}

	// Extract type (default to "object" if not specified)
	if typeVal, exists := paramMap["type"]; exists {
		if typeStr, ok := typeVal.(string); ok {
			result.Type = typeStr
		} else {
			result.Type = "object"
		}
	} else {
		result.Type = "object"
	}

	// Extract properties
	if propsVal, exists := paramMap["properties"]; exists {
		if propsMap, ok := propsVal.(map[string]interface{}); ok {
			result.Properties = propsMap
		}
	}

	// Extract required fields
	if reqVal, exists := paramMap["required"]; exists {
		if reqSlice, ok := reqVal.([]interface{}); ok {
			required := make([]string, 0, len(reqSlice))
			for _, item := range reqSlice {
				if str, ok := item.(string); ok {
					required = append(required, str)
				}
			}
			result.Required = required
		} else if reqStrSlice, ok := reqVal.([]string); ok {
			result.Required = reqStrSlice
		}
	}

	return result
}

// Helper functions

// formatBytes formats byte size in human readable format
func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// contains checks if a string contains a substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > len(substr) && containsHelper(s, substr)))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if toLower(s[i+j]) != toLower(substr[j]) {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func toLower(b byte) byte {
	if b >= 'A' && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

// validateMessageSize validates the total size of a message
func (c *Client) validateMessageSize(totalSize int64) error {
	// Use reasonable defaults for size limits (10MB)
	maxTotalSize := int64(10 * 1024 * 1024)

	if totalSize > maxTotalSize {
		return &llm.Error{
			Code:    "message_size_exceeded",
			Message: fmt.Sprintf("Message size %d bytes exceeds limit %d bytes", totalSize, maxTotalSize),
			Type:    "validation_error",
		}
	}

	return nil
}

// validateImageContent validates image content for DeepSeek compatibility
func (c *Client) validateImageContent(img *llm.ImageContent, modelInfo llm.ModelInfo) error {
	if img == nil {
		return &llm.Error{
			Code:    "invalid_content",
			Message: "Image content cannot be nil",
			Type:    "validation_error",
		}
	}

	// Check if model supports vision
	if !modelInfo.SupportsVision {
		return &llm.Error{
			Code:    "vision_not_supported",
			Message: fmt.Sprintf("Model %s does not support vision/image content", c.model),
			Type:    "validation_error",
		}
	}

	// Validate image size (5MB limit)
	maxImageSize := int64(5 * 1024 * 1024)
	if img.Size() > maxImageSize {
		return &llm.Error{
			Code:    "image_size_exceeded",
			Message: fmt.Sprintf("Image size %d bytes exceeds limit %d bytes", img.Size(), maxImageSize),
			Type:    "validation_error",
		}
	}

	// Basic MIME type validation
	if !isValidImageMimeType(img.MimeType) {
		return &llm.Error{
			Code:    "unsupported_image_type",
			Message: fmt.Sprintf("Image MIME type %s is not supported", img.MimeType),
			Type:    "validation_error",
		}
	}

	// Security validation disabled for now
	// if err := ValidateContentSecurity(img); err != nil {
	//	return &llm.Error{
	//		Code:    "security_validation_failed",
	//		Message: fmt.Sprintf("Image failed security validation: %v", err),
	//		Type:    "validation_error",
	//	}
	// }

	return nil
}

// validateFileContent validates file content for DeepSeek compatibility
func (c *Client) validateFileContent(file *llm.FileContent, modelInfo llm.ModelInfo) error {
	if file == nil {
		return &llm.Error{
			Code:    "invalid_content",
			Message: "File content cannot be nil",
			Type:    "validation_error",
		}
	}

	// Check if model supports files
	if !modelInfo.SupportsFiles {
		return &llm.Error{
			Code:    "files_not_supported",
			Message: fmt.Sprintf("Model %s does not support file content", c.model),
			Type:    "validation_error",
		}
	}

	// Validate file size (10MB limit)
	maxFileSize := int64(10 * 1024 * 1024)
	if file.Size() > maxFileSize {
		return &llm.Error{
			Code:    "file_size_exceeded",
			Message: fmt.Sprintf("File size %d bytes exceeds limit %d bytes", file.Size(), maxFileSize),
			Type:    "validation_error",
		}
	}

	// Basic MIME type validation
	if !isValidFileMimeType(file.MimeType) {
		return &llm.Error{
			Code:    "unsupported_file_type",
			Message: fmt.Sprintf("File MIME type %s is not supported", file.MimeType),
			Type:    "validation_error",
		}
	}

	// Security validation disabled for now
	// if err := ValidateContentSecurity(file); err != nil {
	//	return &llm.Error{
	//		Code:    "security_validation_failed",
	//		Message: fmt.Sprintf("File failed security validation: %v", err),
	//		Type:    "validation_error",
	//	}
	// }

	return nil
}

// convertImageContent converts ImageContent to DeepSeek format
func (c *Client) convertImageContent(img *llm.ImageContent, modelInfo llm.ModelInfo) (string, error) {
	if img == nil {
		return "", fmt.Errorf("image content is nil")
	}

	// Since DeepSeek may not support native multi-modal content in all models,
	// we convert images to text descriptions for now
	// This can be enhanced when DeepSeek adds full multi-modal support

	if img.HasURL() {
		return fmt.Sprintf("[Image: %s, Type: %s]", img.URL, img.MimeType), nil
	} else if img.HasData() {
		sizeStr := formatBytes(img.Size())
		dimensionsStr := ""
		if img.Width > 0 && img.Height > 0 {
			dimensionsStr = fmt.Sprintf(", Dimensions: %dx%d", img.Width, img.Height)
		}

		return fmt.Sprintf("[Image: base64 data, Type: %s, Size: %s%s]",
			img.MimeType, sizeStr, dimensionsStr), nil
	}

	return "[Image: no data available]", nil
}

// convertFileContent converts FileContent to DeepSeek format
func (c *Client) convertFileContent(file *llm.FileContent, modelInfo llm.ModelInfo) (string, error) {
	if file == nil {
		return "", fmt.Errorf("file content is nil")
	}

	// For text-based files, include the actual content
	if file.HasData() {
		switch file.MimeType {
		case "text/plain", "text/csv", "application/json", "text/markdown":
			// Include the actual text content
			content := string(file.Data)
			// Add a header to identify the file
			return fmt.Sprintf("[File: %s (%s)]\n%s", file.Filename, file.MimeType, content), nil

		case "application/pdf":
			// For PDF, provide metadata (actual text extraction would require additional libraries)
			return fmt.Sprintf("[PDF File: %s, Size: %s]",
				file.Filename, formatBytes(file.FileSize)), nil

		default:
			// For other file types, provide metadata
			return fmt.Sprintf("[File: %s, Type: %s, Size: %s]",
				file.Filename, file.MimeType, formatBytes(file.FileSize)), nil
		}
	} else if file.HasURL() {
		// File referenced by URL
		return fmt.Sprintf("[File Reference: %s (%s), Type: %s, Size: %s]",
			file.Filename, file.URL, file.MimeType, formatBytes(file.FileSize)), nil
	}

	return fmt.Sprintf("[File: %s, Type: %s]", file.Filename, file.MimeType), nil
}

// Basic MIME type validation functions
func isValidImageMimeType(mimeType string) bool {
	switch mimeType {
	case "image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func isValidFileMimeType(mimeType string) bool {
	switch mimeType {
	case "text/plain", "text/csv", "application/json", "text/markdown", "application/pdf":
		return true
	default:
		return false
	}
}
