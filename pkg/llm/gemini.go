package llm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"google.golang.org/genai"
)

type GeminiClient struct {
	model    string
	provider string
	genai    *genai.Client
}

// NewGeminiClient creates a new Gemini client using the official Google Generative AI library.
func NewGeminiClient(config ClientConfig) (Client, error) {
	if config.APIKey == "" {
		return nil, &Error{Code: "missing_api_key", Message: "API key is required for Gemini", Type: "authentication_error"}
	}
	if config.Model == "" {
		config.Model = "gemini-1.5-flash"
	}

	// Create genai client config
	genaiConfig := &genai.ClientConfig{
		APIKey:  config.APIKey,
		Backend: genai.BackendGeminiAPI,
	}

	// Set timeout if specified
	if config.Timeout > 0 {
		genaiConfig.HTTPOptions.Timeout = &config.Timeout
	}

	// Create the genai client
	genaiClient, err := genai.NewClient(context.Background(), genaiConfig)
	if err != nil {
		return nil, &Error{
			Code:    "client_creation_error",
			Message: fmt.Sprintf("Failed to create genai client: %v", err),
			Type:    "internal_error",
		}
	}

	return &GeminiClient{
		model:    config.Model,
		provider: "gemini",
		genai:    genaiClient,
	}, nil
}

// ChatCompletion performs a non-streaming content generation request.
func (c *GeminiClient) ChatCompletion(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	// Convert our messages to genai Content format
	contents, err := c.convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// Create generation config
	config := &genai.GenerateContentConfig{}
	if req.Temperature != nil {
		config.Temperature = req.Temperature
	}
	if req.MaxTokens != nil {
		config.MaxOutputTokens = safeIntToInt32(*req.MaxTokens)
	}

	// Create a chat session with history
	var history []*genai.Content
	if len(contents) > 1 {
		// All but the last message are history
		history = contents[:len(contents)-1]
	}

	chat, err := c.genai.Chats.Create(ctx, c.model, config, history)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Get the last message as the current input
	var parts []genai.Part
	if len(contents) > 0 {
		lastContent := contents[len(contents)-1]
		parts = make([]genai.Part, len(lastContent.Parts))
		for i, part := range lastContent.Parts {
			parts[i] = *part
		}
	}

	// Send the message
	response, err := chat.SendMessage(ctx, parts...)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Convert response to our format
	return c.convertResponse(response), nil
}

// convertMessages converts our internal message format to genai Content format
func (c *GeminiClient) convertMessages(messages []Message) ([]*genai.Content, error) {
	var contents []*genai.Content

	for _, msg := range messages {
		role := genai.RoleUser
		if msg.Role == RoleAssistant {
			role = genai.RoleModel
		} else if msg.Role == RoleSystem {
			// System messages are handled as SystemInstruction in config, skip for now
			// TODO: We could handle this by prepending to the first user message
			continue
		}

		var parts []*genai.Part
		for _, content := range msg.Content {
			if text, ok := content.(*TextContent); ok {
				parts = append(parts, genai.NewPartFromText(text.Text))
			} else if img, ok := content.(*ImageContent); ok {
				if img.MimeType != "" && len(img.Data) > 0 {
					parts = append(parts, genai.NewPartFromBytes(img.Data, img.MimeType))
				}
			}
		}

		if len(parts) > 0 {
			contents = append(contents, &genai.Content{
				Role:  role,
				Parts: parts,
			})
		}
	}

	if len(contents) == 0 {
		return nil, &Error{Code: "invalid_request", Message: "No valid messages provided", Type: "validation_error", StatusCode: 400}
	}

	return contents, nil
}

// convertResponse converts genai response to our internal format
func (c *GeminiClient) convertResponse(resp *genai.GenerateContentResponse) *ChatResponse {
	if len(resp.Candidates) == 0 {
		return &ChatResponse{
			ID:      fmt.Sprintf("gemini-%s", time.Now().Format(time.RFC3339Nano)),
			Model:   c.model,
			Choices: []Choice{},
		}
	}

	candidate := resp.Candidates[0]
	text := candidate.Content.Parts[0].Text

	finishReason := "stop"
	if candidate.FinishReason == genai.FinishReasonMaxTokens {
		finishReason = "length"
	} else if strings.Contains(string(candidate.FinishReason), "SAFETY") {
		finishReason = "content_filter"
	}

	message := Message{
		Role:    RoleAssistant,
		Content: []MessageContent{NewTextContent(text)},
	}

	choice := Choice{
		Index:        0,
		Message:      message,
		FinishReason: finishReason,
	}

	return &ChatResponse{
		ID:      fmt.Sprintf("gemini-%s", time.Now().Format(time.RFC3339Nano)),
		Model:   c.model,
		Choices: []Choice{choice},
	}
}

// convertError converts genai errors to our internal error format
func (c *GeminiClient) convertError(err error) *Error {
	if err == nil {
		return nil
	}

	// Check if it's already our error type
	if ourErr, ok := err.(*Error); ok {
		return ourErr
	}

	// Convert specific error types based on error message/content
	errMsg := err.Error()

	// Check for authentication errors
	if strings.Contains(errMsg, "API key") ||
		strings.Contains(errMsg, "authentication") ||
		strings.Contains(errMsg, "unauthorized") ||
		strings.Contains(errMsg, "401") {
		return &Error{
			Code:       "authentication_error",
			Message:    errMsg,
			Type:       "authentication_error",
			StatusCode: 401,
		}
	}

	// Check for rate limiting errors
	if strings.Contains(errMsg, "rate limit") ||
		strings.Contains(errMsg, "429") {
		return &Error{
			Code:       "rate_limit_error",
			Message:    errMsg,
			Type:       "rate_limit_error",
			StatusCode: 429,
		}
	}

	// Check for quota errors
	if strings.Contains(errMsg, "quota") ||
		strings.Contains(errMsg, "403") {
		return &Error{
			Code:       "quota_error",
			Message:    errMsg,
			Type:       "quota_error",
			StatusCode: 403,
		}
	}

	// Default error conversion
	return &Error{
		Code:    "api_error",
		Message: errMsg,
		Type:    "api_error",
	}
}

func (c *GeminiClient) StreamChatCompletion(ctx context.Context, req ChatRequest) (<-chan StreamEvent, error) {
	// Convert our messages to genai Content format
	contents, err := c.convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// Create generation config
	config := &genai.GenerateContentConfig{}
	if req.Temperature != nil {
		config.Temperature = req.Temperature
	}
	if req.MaxTokens != nil {
		config.MaxOutputTokens = safeIntToInt32(*req.MaxTokens)
	}

	// Create a chat session with history
	var history []*genai.Content
	if len(contents) > 1 {
		// All but the last message are history
		history = contents[:len(contents)-1]
	}

	chat, err := c.genai.Chats.Create(ctx, c.model, config, history)
	if err != nil {
		return nil, c.convertError(err)
	}

	// Get the last message as the current input
	var parts []genai.Part
	if len(contents) > 0 {
		lastContent := contents[len(contents)-1]
		parts = make([]genai.Part, len(lastContent.Parts))
		for i, part := range lastContent.Parts {
			parts[i] = *part
		}
	}

	// Create output channel
	ch := make(chan StreamEvent)

	go func() {
		defer close(ch)

		// Send streaming message
		for response, err := range chat.SendMessageStream(ctx, parts...) {
			if err != nil {
				ch <- NewErrorEvent(c.convertError(err))
				return
			}

			// Convert response to delta
			if len(response.Candidates) > 0 && len(response.Candidates[0].Content.Parts) > 0 {
				text := response.Candidates[0].Content.Parts[0].Text
				if text != "" {
					delta := &MessageDelta{Content: []MessageContent{NewTextContent(text)}}
					ch <- NewDeltaEvent(0, delta)
				}
			}
		}

		// Send done event
		ch <- NewDoneEvent(0, "stop")
	}()

	return ch, nil
}

func (c *GeminiClient) GetModelInfo() ModelInfo {
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

func (c *GeminiClient) Close() error {
	// The genai client doesn't provide a Close method, so we don't need to do anything
	return nil
}
