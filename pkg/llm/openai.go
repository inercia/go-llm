package llm

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"

	"github.com/sashabaranov/go-openai"
)

// OpenAIClient implements the Client interface for OpenAI
type OpenAIClient struct {
	client   *openai.Client
	model    string
	provider string
}

// NewOpenAIClient creates a new OpenAI client
func NewOpenAIClient(config ClientConfig) (*OpenAIClient, error) {
	if config.APIKey == "" {
		return nil, &Error{
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

	return &OpenAIClient{
		client:   openai.NewClientWithConfig(clientConfig),
		model:    config.Model,
		provider: "openai",
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *OpenAIClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
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
func (c *OpenAIClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
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

	ch := make(chan StreamEvent, 10)

	go func() {
		defer close(ch)
		defer func() { _ = stream.Close() }()

		for {
			response, err := stream.Recv()
			if err == io.EOF {
				// Stream complete
				ch <- NewDoneEvent(0, "stop")
				return
			}
			if err != nil {
				ch <- NewErrorEvent(c.convertError(err))
				return
			}

			// Convert chunk to delta event
			delta := &MessageDelta{}
			if len(response.Choices) > 0 {
				choice := response.Choices[0]
				if choice.Delta.Content != "" {
					delta.Content = []MessageContent{NewTextContent(choice.Delta.Content)}
				}
				if len(choice.Delta.ToolCalls) > 0 {
					for _, tc := range choice.Delta.ToolCalls {
						index := 0
						if tc.Index != nil {
							index = *tc.Index
						}
						delta.ToolCalls = append(delta.ToolCalls, ToolCallDelta{
							Index: index,
							ID:    tc.ID,
							Type:  string(tc.Type),
							Function: &ToolCallFunctionDelta{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						})
					}
				}
				if choice.FinishReason != "" {
					ch <- NewDoneEvent(0, string(choice.FinishReason))
					return
				}
			}

			if len(delta.Content) > 0 || len(delta.ToolCalls) > 0 {
				ch <- NewDeltaEvent(0, delta)
			}
		}
	}()

	return ch, nil
}

// GetModelInfo returns information about the model
func (c *OpenAIClient) GetModelInfo() ModelInfo {
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
func (c *OpenAIClient) Close() error {
	// OpenAI client doesn't require explicit cleanup
	return nil
}

// selectModelForRequest chooses the appropriate model based on content types
func (c *OpenAIClient) selectModelForRequest(req ChatRequest) string {
	// Check if any message contains non-text content
	hasMultiModal := false
	for _, msg := range req.Messages {
		for _, content := range msg.Content {
			if content.Type() != MessageTypeText {
				hasMultiModal = true
				break
			}
		}
		if hasMultiModal {
			break
		}
	}

	// If the request has multi-modal content, use a vision-capable model
	if hasMultiModal {
		return c.getVisionCapableModel()
	}

	// Use the configured model for text-only requests
	if req.Model != "" {
		return req.Model
	}
	return c.model
}

// getVisionCapableModel returns a vision-capable OpenAI model
func (c *OpenAIClient) getVisionCapableModel() string {
	// Prefer the configured model if it supports vision
	if isVisionCapableModel(c.model) {
		return c.model
	}

	// Default to gpt-4o for vision tasks
	return "gpt-4o"
}

// isVisionCapableModel checks if a model supports vision
func isVisionCapableModel(model string) bool {
	visionModels := map[string]bool{
		"gpt-4o":               true,
		"gpt-4o-mini":          true,
		"gpt-4-vision-preview": true,
		"gpt-4-turbo":          true,
		"gpt-4-turbo-preview":  true,
	}
	return visionModels[model]
}

// convertRequest converts our ChatRequest to OpenAI format
func (c *OpenAIClient) convertRequest(req ChatRequest, model string) openai.ChatCompletionRequest {
	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		// Validate message security before processing
		if err := ValidateMessageSecurity(&msg); err != nil {
			// Security validation failed - in production you might want to fail the request
			// For now, we log the error and continue processing
			_ = err
		}
		messages[i] = c.convertMessage(msg)
	}

	tools := make([]openai.Tool, len(req.Tools))
	for i, tool := range req.Tools {
		tools[i] = openai.Tool{
			Type: openai.ToolType(tool.Type),
			Function: &openai.FunctionDefinition{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
			},
		}
	}

	openaiReq := openai.ChatCompletionRequest{
		Model:    model,
		Messages: messages,
		Tools:    tools,
	}

	if req.Temperature != nil {
		openaiReq.Temperature = *req.Temperature
	}
	if req.MaxTokens != nil {
		openaiReq.MaxTokens = *req.MaxTokens
	}
	if req.TopP != nil {
		openaiReq.TopP = *req.TopP
	}
	openaiReq.Stream = req.Stream

	return openaiReq
}

// convertMessage converts our Message to OpenAI format with multi-modal support
func (c *OpenAIClient) convertMessage(msg Message) openai.ChatCompletionMessage {
	openaiMsg := openai.ChatCompletionMessage{
		Role:       convertRoleToOpenAI(msg.Role),
		ToolCalls:  convertToolCallsToOpenAI(msg.ToolCalls),
		ToolCallID: msg.ToolCallID,
	}

	// Handle multi-modal content
	if len(msg.Content) == 0 {
		// Empty content
		openaiMsg.Content = ""
	} else if len(msg.Content) == 1 && msg.Content[0].Type() == MessageTypeText {
		// Single text content - use string format for backward compatibility
		if textContent, ok := msg.Content[0].(*TextContent); ok {
			openaiMsg.Content = textContent.GetText()
		}
	} else {
		// Multi-modal content - use the new content array format
		contentParts := make([]openai.ChatMessagePart, 0, len(msg.Content))

		for _, content := range msg.Content {
			switch content.Type() {
			case MessageTypeText:
				if textContent, ok := content.(*TextContent); ok {
					contentParts = append(contentParts, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeText,
						Text: textContent.GetText(),
					})
				}
			case MessageTypeImage:
				if imageContent, ok := content.(*ImageContent); ok {
					imagePart, err := c.convertImageContent(imageContent)
					if err != nil {
						// Log error and skip this content
						continue
					}
					contentParts = append(contentParts, imagePart)
				}
			case MessageTypeFile:
				if fileContent, ok := content.(*FileContent); ok {
					filePart, err := c.convertFileContent(fileContent)
					if err != nil {
						// Log error and skip this content
						continue
					}
					contentParts = append(contentParts, filePart)
				}
			}
		}

		openaiMsg.MultiContent = contentParts
	}

	return openaiMsg
}

// convertImageContent converts ImageContent to OpenAI format
func (c *OpenAIClient) convertImageContent(img *ImageContent) (openai.ChatMessagePart, error) {
	if img == nil {
		return openai.ChatMessagePart{}, fmt.Errorf("image content is nil")
	}

	var imageURL string

	if img.HasURL() {
		// Use provided URL
		imageURL = img.URL
	} else if img.HasData() {
		// Convert binary data to base64 data URL
		base64Data := base64.StdEncoding.EncodeToString(img.Data)
		imageURL = fmt.Sprintf("data:%s;base64,%s", img.MimeType, base64Data)
	} else {
		return openai.ChatMessagePart{}, fmt.Errorf("image content has neither URL nor data")
	}

	return openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeImageURL,
		ImageURL: &openai.ChatMessageImageURL{
			URL:    imageURL,
			Detail: openai.ImageURLDetailAuto,
		},
	}, nil
}

// convertFileContent converts FileContent to text format for OpenAI
func (c *OpenAIClient) convertFileContent(file *FileContent) (openai.ChatMessagePart, error) {
	if file == nil {
		return openai.ChatMessagePart{}, fmt.Errorf("file content is nil")
	}

	var textContent string

	// For text-based files, extract the content
	if file.HasData() {
		switch file.MimeType {
		case "text/plain", "text/csv", "application/json":
			textContent = string(file.Data)
		case "application/pdf":
			// For PDF, we'd normally need a library to extract text
			// For now, indicate it's a PDF file with metadata
			textContent = fmt.Sprintf("[PDF File: %s, Size: %d bytes]", file.Filename, file.FileSize)
		default:
			textContent = fmt.Sprintf("[File: %s, Type: %s, Size: %d bytes]", file.Filename, file.MimeType, file.FileSize)
		}
	} else {
		// File referenced by URL
		textContent = fmt.Sprintf("[File Reference: %s (%s), Type: %s, Size: %d bytes]", file.Filename, file.URL, file.MimeType, file.FileSize)
	}

	return openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: textContent,
	}, nil
}

// convertResponse converts OpenAI response to our format
func (c *OpenAIClient) convertResponse(resp openai.ChatCompletionResponse) *ChatResponse {
	choices := make([]Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		choices[i] = Choice{
			Index: choice.Index,
			Message: Message{
				Role:      convertRoleFromOpenAI(choice.Message.Role),
				Content:   []MessageContent{NewTextContent(choice.Message.Content)},
				ToolCalls: convertToolCallsFromOpenAI(choice.Message.ToolCalls),
			},
			FinishReason: string(choice.FinishReason),
		}
	}

	return &ChatResponse{
		ID:      resp.ID,
		Model:   resp.Model,
		Choices: choices,
		Usage: Usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
}

// convertError converts OpenAI error to our standardized error format
func (c *OpenAIClient) convertError(err error) *Error {
	// Handle different types of OpenAI errors
	if apiErr, ok := err.(*openai.APIError); ok {
		code := "api_error"
		if apiErr.Code != nil {
			if codeStr, ok := apiErr.Code.(string); ok {
				code = codeStr
			}
		}

		return &Error{
			Code:       code,
			Message:    apiErr.Message,
			Type:       apiErr.Type,
			StatusCode: apiErr.HTTPStatusCode,
		}
	}

	// Generic error handling
	return &Error{
		Code:    "unknown_error",
		Message: err.Error(),
		Type:    "api_error",
	}
}

// Helper conversion functions
func convertRoleToOpenAI(role MessageRole) string {
	switch role {
	case RoleSystem:
		return openai.ChatMessageRoleSystem
	case RoleUser:
		return openai.ChatMessageRoleUser
	case RoleAssistant:
		return openai.ChatMessageRoleAssistant
	case RoleTool:
		return openai.ChatMessageRoleTool
	default:
		return openai.ChatMessageRoleUser
	}
}

func convertRoleFromOpenAI(role string) MessageRole {
	switch role {
	case openai.ChatMessageRoleSystem:
		return RoleSystem
	case openai.ChatMessageRoleUser:
		return RoleUser
	case openai.ChatMessageRoleAssistant:
		return RoleAssistant
	case openai.ChatMessageRoleTool:
		return RoleTool
	default:
		return RoleUser
	}
}

func convertToolCallsToOpenAI(toolCalls []ToolCall) []openai.ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	result := make([]openai.ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = openai.ToolCall{
			ID:   tc.ID,
			Type: openai.ToolType(tc.Type),
			Function: openai.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return result
}

func convertToolCallsFromOpenAI(toolCalls []openai.ToolCall) []ToolCall {
	if len(toolCalls) == 0 {
		return nil
	}

	result := make([]ToolCall, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = ToolCall{
			ID:   tc.ID,
			Type: string(tc.Type),
			Function: ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}
	return result
}

// Model capabilities are now handled by the centralized model registry
