package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/inercia/go-llm/pkg/llm"
)

// safeIntToInt32 safely converts int to int32
func safeIntToInt32(val int) int32 {
	if val > 2147483647 {
		return 2147483647
	}
	if val < -2147483648 {
		return -2147483648
	}
	return int32(val)
}

// modelCapabilities defines the capabilities for a model pattern
type modelCapabilities struct {
	pattern        *regexp.Regexp
	maxTokens      int
	supportsTools  bool
	supportsVision bool
	supportsFiles  bool
}

// modelCapabilitiesList defines capabilities for different Gemini models
// Models are matched in order, first match wins
var modelCapabilitiesList = []modelCapabilities{
	// Gemini 1.5 Pro models (2M context)
	{
		pattern:        regexp.MustCompile(`gemini-1\.5-pro`),
		maxTokens:      2000000,
		supportsTools:  true,
		supportsVision: true,
		supportsFiles:  true,
	},
	// Gemini 1.5 Flash models (1M context)
	{
		pattern:        regexp.MustCompile(`gemini-1\.5-flash`),
		maxTokens:      1000000,
		supportsTools:  true,
		supportsVision: true,
		supportsFiles:  true,
	},
	// Gemini 1.0 Pro Vision
	{
		pattern:        regexp.MustCompile(`gemini-.*-vision`),
		maxTokens:      30720,
		supportsTools:  true,
		supportsVision: true,
		supportsFiles:  true,
	},
}

type Client struct {
	model    string
	provider string
	genai    *genai.Client

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new Gemini client using the official Google Generative AI library.
func NewClient(config llm.ClientConfig) (*Client, error) {
	if config.APIKey == "" {
		return nil, &llm.Error{Code: "missing_api_key", Message: "API key is required for Gemini", Type: "authentication_error"}
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
		return nil, &llm.Error{
			Code:    "client_creation_error",
			Message: fmt.Sprintf("Failed to create genai client: %v", err),
			Type:    "internal_error",
		}
	}

	return &Client{
		model:    config.Model,
		provider: "gemini",
		genai:    genaiClient,
	}, nil
}

// ChatCompletion performs a non-streaming content generation request.
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
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

	// Handle ResponseFormat by adding instructions to the system message
	// Gemini doesn't support structured outputs natively, so we use prompt engineering
	if req.ResponseFormat != nil {
		contents = c.addResponseFormatInstructions(contents, req.ResponseFormat)
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
func (c *Client) convertMessages(messages []llm.Message) ([]*genai.Content, error) {
	var contents []*genai.Content

	for _, msg := range messages {
		role := genai.RoleUser
		if msg.Role == llm.RoleAssistant {
			role = genai.RoleModel
		} else if msg.Role == llm.RoleSystem {
			// System messages are handled as SystemInstruction in config, skip for now
			continue
		}

		var parts []*genai.Part
		for _, content := range msg.Content {
			if text, ok := content.(*llm.TextContent); ok {
				parts = append(parts, genai.NewPartFromText(text.GetText()))
			} else if img, ok := content.(*llm.ImageContent); ok {
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
		return nil, &llm.Error{Code: "invalid_request", Message: "No valid messages provided", Type: "validation_error", StatusCode: 400}
	}

	return contents, nil
}

// convertResponse converts genai response to our internal format
func (c *Client) convertResponse(resp *genai.GenerateContentResponse) *llm.ChatResponse {
	if len(resp.Candidates) == 0 {
		return &llm.ChatResponse{
			ID:      fmt.Sprintf("gemini-%s", time.Now().Format(time.RFC3339Nano)),
			Model:   c.model,
			Choices: []llm.Choice{},
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

	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: []llm.MessageContent{llm.NewTextContent(text)},
	}

	choice := llm.Choice{
		Index:        0,
		Message:      message,
		FinishReason: finishReason,
	}

	return &llm.ChatResponse{
		ID:      fmt.Sprintf("gemini-%s", time.Now().Format(time.RFC3339Nano)),
		Model:   c.model,
		Choices: []llm.Choice{choice},
	}
}

// convertError converts genai errors to our internal error format
func (c *Client) convertError(err error) *llm.Error {
	if err == nil {
		return nil
	}

	// Check if it's already our error type
	if ourErr, ok := err.(*llm.Error); ok {
		return ourErr
	}

	// Convert specific error types based on error message/content
	errMsg := err.Error()

	// Check for authentication errors
	if strings.Contains(errMsg, "API key") ||
		strings.Contains(errMsg, "authentication") ||
		strings.Contains(errMsg, "unauthorized") ||
		strings.Contains(errMsg, "401") {
		return &llm.Error{
			Code:       "authentication_error",
			Message:    errMsg,
			Type:       "authentication_error",
			StatusCode: 401,
		}
	}

	// Check for rate limiting errors
	if strings.Contains(errMsg, "rate limit") ||
		strings.Contains(errMsg, "429") {
		return &llm.Error{
			Code:       "rate_limit_error",
			Message:    errMsg,
			Type:       "rate_limit_error",
			StatusCode: 429,
		}
	}

	// Check for quota errors
	if strings.Contains(errMsg, "quota") ||
		strings.Contains(errMsg, "403") {
		return &llm.Error{
			Code:       "quota_error",
			Message:    errMsg,
			Type:       "quota_error",
			StatusCode: 403,
		}
	}

	// Default error conversion
	return &llm.Error{
		Code:    "api_error",
		Message: errMsg,
		Type:    "api_error",
	}
}

func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
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
	ch := make(chan llm.StreamEvent)

	go func() {
		defer close(ch)

		// Send streaming message
		for response, err := range chat.SendMessageStream(ctx, parts...) {
			if err != nil {
				ch <- llm.NewErrorEvent(c.convertError(err))
				return
			}

			// Convert response to delta
			if len(response.Candidates) > 0 && len(response.Candidates[0].Content.Parts) > 0 {
				text := response.Candidates[0].Content.Parts[0].Text
				if text != "" {
					delta := &llm.MessageDelta{Content: []llm.MessageContent{llm.NewTextContent(text)}}
					ch <- llm.NewDeltaEvent(0, delta)
				}
			}
		}

		// Send done event
		ch <- llm.NewDoneEvent(0, "stop")
	}()

	return ch, nil
}

// addResponseFormatInstructions adds JSON formatting instructions to the content when ResponseFormat is specified
// Since Gemini doesn't support structured outputs natively, we use prompt engineering
func (c *Client) addResponseFormatInstructions(contents []*genai.Content, responseFormat *llm.ResponseFormat) []*genai.Content {
	if responseFormat == nil {
		return contents
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
		return contents // No formatting needed for text responses
	}

	// Add the instruction as a system message at the beginning
	systemContent := &genai.Content{
		Role: "user", // Gemini uses "user" for system-like instructions
		Parts: []*genai.Part{
			{Text: instruction},
		},
	}

	// Prepend the system instruction
	return append([]*genai.Content{systemContent}, contents...)
}

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "gemini",
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

// performHealthCheck performs a simple health check on the Gemini API
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create simple generation config
	config := &genai.GenerateContentConfig{
		MaxOutputTokens: 1,
	}

	// Create a simple chat session
	chat, err := c.genai.Chats.Create(ctx, c.model, config, nil)
	if err != nil {
		return false
	}

	// Try a simple message
	_, err = chat.SendMessage(ctx, *genai.NewPartFromText("test"))
	return err == nil
}

func (c *Client) GetModelInfo() llm.ModelInfo {
	// Default capabilities
	caps := modelCapabilities{
		maxTokens:      30720,
		supportsTools:  true,
		supportsVision: true,
		supportsFiles:  true,
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
		Provider:          c.provider,
		MaxTokens:         caps.maxTokens,
		SupportsTools:     caps.supportsTools,
		SupportsVision:    caps.supportsVision,
		SupportsFiles:     caps.supportsFiles,
		SupportsStreaming: true,
	}
}

func (c *Client) Close() error {
	// The genai client doesn't provide a Close method, so we don't need to do anything
	return nil
}
