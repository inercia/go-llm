package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/inercia/go-llm/pkg/llm"
)

// Client implements the llm.Client interface for AWS Bedrock
type Client struct {
	bedrockClient        *bedrock.Client
	bedrockRuntimeClient *bedrockruntime.Client
	model                string
	region               string
	provider             string

	// Health check caching
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new AWS Bedrock client
func NewClient(config llm.ClientConfig) (*Client, error) {
	// Get region from Extra config or use default
	region := "us-east-1"
	if config.Extra != nil {
		if r, exists := config.Extra["region"]; exists {
			region = r
		}
	}

	// Create AWS configuration
	awsConfig, err := awsconfig.LoadDefaultConfig(context.Background(), awsconfig.WithRegion(region))
	if err != nil {
		return nil, &llm.Error{
			Code:    "aws_config_error",
			Message: fmt.Sprintf("Failed to load AWS configuration: %v", err),
			Type:    "authentication_error",
		}
	}

	// Create Bedrock clients with optional custom endpoints
	bedrockClient := bedrock.NewFromConfig(awsConfig, func(o *bedrock.Options) {
		if config.Extra != nil {
			if endpoint, exists := config.Extra["bedrock_endpoint"]; exists && endpoint != "" {
				o.BaseEndpoint = aws.String(endpoint)
			}
		}
	})

	bedrockRuntimeClient := bedrockruntime.NewFromConfig(awsConfig, func(o *bedrockruntime.Options) {
		if config.Extra != nil {
			if endpoint, exists := config.Extra["bedrock_runtime_endpoint"]; exists && endpoint != "" {
				o.BaseEndpoint = aws.String(endpoint)
			}
			// Support BaseURL for backward compatibility and consistency with other providers
			if endpoint, exists := config.Extra["base_url"]; exists && endpoint != "" {
				o.BaseEndpoint = aws.String(endpoint)
			}
		}
		// Support config.BaseURL for consistency with other providers
		if config.BaseURL != "" {
			o.BaseEndpoint = aws.String(config.BaseURL)
		}
	})

	return &Client{
		bedrockClient:        bedrockClient,
		bedrockRuntimeClient: bedrockRuntimeClient,
		model:                config.Model,
		region:               region,
		provider:             "bedrock",
	}, nil
}

// ChatCompletion performs a chat completion request
func (c *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Convert request based on model type
	payload, err := c.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Invoke model
	response, err := c.bedrockRuntimeClient.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(c.model),
		ContentType: aws.String("application/json"),
		Body:        payload,
	})
	if err != nil {
		return nil, c.convertError(err)
	}

	// Convert response back to our format
	return c.convertResponse(response.Body)
}

// StreamChatCompletion performs a streaming chat completion request
func (c *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Convert request based on model type
	payload, err := c.convertRequest(req)
	if err != nil {
		return nil, err
	}

	// Invoke model with streaming
	response, err := c.bedrockRuntimeClient.InvokeModelWithResponseStream(ctx, &bedrockruntime.InvokeModelWithResponseStreamInput{
		ModelId:     aws.String(c.model),
		ContentType: aws.String("application/json"),
		Body:        payload,
	})
	if err != nil {
		return nil, c.convertError(err)
	}

	ch := make(chan llm.StreamEvent, 10)

	go func() {
		defer close(ch)

		for event := range response.GetStream().Events() {
			switch v := event.(type) {
			case *types.ResponseStreamMemberChunk:
				// Parse the chunk and send delta event
				if err := c.processStreamChunk(v.Value.Bytes, ch); err != nil {
					ch <- llm.NewErrorEvent(c.convertError(err))
					return
				}
			case *types.UnknownUnionMember:
				// Log unknown events but continue processing
				continue
			default:
				// Unknown event type, continue processing
				continue
			}
		}

		// Send completion event
		ch <- llm.NewDoneEvent(0, "stop")
	}()

	return ch, nil
}

// GetRemote returns information about the remote client
func (c *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "bedrock",
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

// performHealthCheck performs a simple health check on AWS Bedrock
func (c *Client) performHealthCheck() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Try to list foundation models as a health check
	_, err := c.bedrockClient.ListFoundationModels(ctx, &bedrock.ListFoundationModelsInput{})
	return err == nil
}

// GetModelInfo returns information about the model being used
func (c *Client) GetModelInfo() llm.ModelInfo {
	maxTokens := c.getMaxTokensForModel(c.model)

	return llm.ModelInfo{
		Name:              c.model,
		Provider:          c.provider,
		MaxTokens:         maxTokens,
		SupportsTools:     c.supportsTools(c.model),
		SupportsVision:    c.supportsVision(c.model),
		SupportsFiles:     c.supportsFiles(c.model),
		SupportsStreaming: true,
	}
}

// Close cleans up any resources used by the client
func (c *Client) Close() error {
	// AWS SDK clients don't require explicit cleanup
	return nil
}

// convertRequest converts our ChatRequest to the appropriate format based on model
func (c *Client) convertRequest(req llm.ChatRequest) ([]byte, error) {
	if c.isClaudeModel() {
		return c.convertToClaudeRequest(req)
	} else if c.isTitanModel() {
		return c.convertToTitanRequest(req)
	} else if c.isLlamaModel() {
		return c.convertToLlamaRequest(req)
	}

	// Default to Claude format for unknown models
	return c.convertToClaudeRequest(req)
}

// convertToClaudeRequest converts to Claude's request format
func (c *Client) convertToClaudeRequest(req llm.ChatRequest) ([]byte, error) {
	// For legacy Claude models (v2), use the prompt format
	if strings.Contains(c.model, "claude-v2") || strings.Contains(c.model, "claude-instant") {
		// Convert messages to Claude prompt format
		prompt := c.messagesToClaudePrompt(req.Messages)

		claudeReq := map[string]interface{}{
			"prompt":               prompt,
			"max_tokens_to_sample": 1000, // Default
		}

		if req.MaxTokens != nil {
			claudeReq["max_tokens_to_sample"] = *req.MaxTokens
		}
		if req.Temperature != nil {
			claudeReq["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			claudeReq["top_p"] = *req.TopP
		}

		return json.Marshal(claudeReq)
	}

	// For newer Claude models (3.x), use the messages API format
	return c.convertToClaudeMessagesRequest(req)
}

// convertToClaudeMessagesRequest converts to Claude 3.x messages format
func (c *Client) convertToClaudeMessagesRequest(req llm.ChatRequest) ([]byte, error) {
	claudeReq := map[string]interface{}{
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens":        1000, // Default
	}

	if req.MaxTokens != nil {
		claudeReq["max_tokens"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		claudeReq["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		claudeReq["top_p"] = *req.TopP
	}

	// Convert messages
	var messages []map[string]interface{}
	var systemMessage string

	for _, msg := range req.Messages {
		if msg.Role == llm.RoleSystem {
			// Collect system messages
			systemMessage += msg.GetText() + "\n"
			continue
		}

		role := string(msg.Role)
		switch msg.Role {
		case llm.RoleAssistant:
			role = "assistant"
		case llm.RoleUser:
			role = "user"
		}

		claudeMsg := map[string]interface{}{
			"role": role,
		}

		// Handle content
		if msg.IsTextOnly() {
			claudeMsg["content"] = msg.GetText()
		} else {
			// Multi-modal content
			var content []map[string]interface{}
			for _, msgContent := range msg.Content {
				switch msgContent.Type() {
				case llm.MessageTypeText:
					if textContent, ok := msgContent.(*llm.TextContent); ok {
						content = append(content, map[string]interface{}{
							"type": "text",
							"text": textContent.GetText(),
						})
					}
				case llm.MessageTypeImage:
					if imgContent, ok := msgContent.(*llm.ImageContent); ok {
						content = append(content, map[string]interface{}{
							"type": "image",
							"source": map[string]interface{}{
								"type":       "base64",
								"media_type": imgContent.MimeType,
								"data":       imgContent.Data,
							},
						})
					}
				}
			}
			claudeMsg["content"] = content
		}

		messages = append(messages, claudeMsg)
	}

	claudeReq["messages"] = messages

	if strings.TrimSpace(systemMessage) != "" {
		claudeReq["system"] = strings.TrimSpace(systemMessage)
	}

	return json.Marshal(claudeReq)
}

// messagesToClaudePrompt converts messages to Claude v2 prompt format
func (c *Client) messagesToClaudePrompt(messages []llm.Message) string {
	var prompt strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleSystem:
			prompt.WriteString(msg.GetText() + "\n\n")
		case llm.RoleUser:
			prompt.WriteString(fmt.Sprintf("\n\nHuman: %s", msg.GetText()))
		case llm.RoleAssistant:
			prompt.WriteString(fmt.Sprintf("\n\nAssistant: %s", msg.GetText()))
		}
	}

	// Ensure we end with Assistant prompt
	if !strings.HasSuffix(prompt.String(), "\n\nAssistant:") {
		prompt.WriteString("\n\nAssistant:")
	}

	return prompt.String()
}

// convertToTitanRequest converts to Amazon Titan request format
func (c *Client) convertToTitanRequest(req llm.ChatRequest) ([]byte, error) {
	// Combine all messages into a single prompt for Titan
	var prompt strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case llm.RoleSystem:
			prompt.WriteString(msg.GetText() + "\n\n")
		case llm.RoleUser:
			prompt.WriteString(fmt.Sprintf("User: %s\n", msg.GetText()))
		case llm.RoleAssistant:
			prompt.WriteString(fmt.Sprintf("Bot: %s\n", msg.GetText()))
		}
	}

	titanReq := map[string]interface{}{
		"inputText": prompt.String(),
		"textGenerationConfig": map[string]interface{}{
			"maxTokenCount": 1000, // Default
			"temperature":   0.7,  // Default
		},
	}

	config := titanReq["textGenerationConfig"].(map[string]interface{})
	if req.MaxTokens != nil {
		config["maxTokenCount"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		config["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		config["topP"] = *req.TopP
	}

	return json.Marshal(titanReq)
}

// convertToLlamaRequest converts to Llama request format
func (c *Client) convertToLlamaRequest(req llm.ChatRequest) ([]byte, error) {
	// Combine messages into prompt for Llama
	var prompt strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case llm.RoleSystem:
			prompt.WriteString(fmt.Sprintf("<s>[INST] <<SYS>>\n%s\n<</SYS>>\n\n", msg.GetText()))
		case llm.RoleUser:
			prompt.WriteString(fmt.Sprintf("%s [/INST]", msg.GetText()))
		case llm.RoleAssistant:
			prompt.WriteString(fmt.Sprintf(" %s </s><s>[INST] ", msg.GetText()))
		}
	}

	llamaReq := map[string]interface{}{
		"prompt": prompt.String(),
	}

	if req.MaxTokens != nil {
		llamaReq["max_gen_len"] = *req.MaxTokens
	}
	if req.Temperature != nil {
		llamaReq["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		llamaReq["top_p"] = *req.TopP
	}

	return json.Marshal(llamaReq)
}

// convertResponse converts the response based on model type
func (c *Client) convertResponse(body []byte) (*llm.ChatResponse, error) {
	if c.isClaudeModel() {
		return c.convertClaudeResponse(body)
	} else if c.isTitanModel() {
		return c.convertTitanResponse(body)
	} else if c.isLlamaModel() {
		return c.convertLlamaResponse(body)
	}

	// Default to Claude
	return c.convertClaudeResponse(body)
}

// convertClaudeResponse converts Claude response format
func (c *Client) convertClaudeResponse(body []byte) (*llm.ChatResponse, error) {
	var claudeResp map[string]interface{}
	if err := json.Unmarshal(body, &claudeResp); err != nil {
		return nil, c.convertError(err)
	}

	var text string
	if completion, ok := claudeResp["completion"].(string); ok {
		// Claude v2 format
		text = completion
	} else if content, ok := claudeResp["content"].([]interface{}); ok {
		// Claude 3.x format
		for _, item := range content {
			if contentItem, ok := item.(map[string]interface{}); ok {
				if contentType, ok := contentItem["type"].(string); ok && contentType == "text" {
					if textContent, ok := contentItem["text"].(string); ok {
						text += textContent
					}
				}
			}
		}
	}

	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: []llm.MessageContent{llm.NewTextContent(text)},
	}

	choice := llm.Choice{
		Index:        0,
		Message:      message,
		FinishReason: "stop",
	}

	return &llm.ChatResponse{
		ID:      fmt.Sprintf("bedrock-%s", time.Now().Format(time.RFC3339Nano)),
		Model:   c.model,
		Choices: []llm.Choice{choice},
	}, nil
}

// convertTitanResponse converts Titan response format
func (c *Client) convertTitanResponse(body []byte) (*llm.ChatResponse, error) {
	var titanResp map[string]interface{}
	if err := json.Unmarshal(body, &titanResp); err != nil {
		return nil, c.convertError(err)
	}

	var text string
	if results, ok := titanResp["results"].([]interface{}); ok && len(results) > 0 {
		if result, ok := results[0].(map[string]interface{}); ok {
			if outputText, ok := result["outputText"].(string); ok {
				text = outputText
			}
		}
	}

	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: []llm.MessageContent{llm.NewTextContent(text)},
	}

	choice := llm.Choice{
		Index:        0,
		Message:      message,
		FinishReason: "stop",
	}

	return &llm.ChatResponse{
		ID:      fmt.Sprintf("bedrock-%s", time.Now().Format(time.RFC3339Nano)),
		Model:   c.model,
		Choices: []llm.Choice{choice},
	}, nil
}

// convertLlamaResponse converts Llama response format
func (c *Client) convertLlamaResponse(body []byte) (*llm.ChatResponse, error) {
	var llamaResp map[string]interface{}
	if err := json.Unmarshal(body, &llamaResp); err != nil {
		return nil, c.convertError(err)
	}

	var text string
	if generation, ok := llamaResp["generation"].(string); ok {
		text = generation
	}

	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: []llm.MessageContent{llm.NewTextContent(text)},
	}

	choice := llm.Choice{
		Index:        0,
		Message:      message,
		FinishReason: "stop",
	}

	return &llm.ChatResponse{
		ID:      fmt.Sprintf("bedrock-%s", time.Now().Format(time.RFC3339Nano)),
		Model:   c.model,
		Choices: []llm.Choice{choice},
	}, nil
}

// processStreamChunk processes a streaming chunk and sends appropriate events
func (c *Client) processStreamChunk(chunkData []byte, ch chan<- llm.StreamEvent) error {
	if c.isClaudeModel() {
		return c.processClaudeStreamChunk(chunkData, ch)
	} else if c.isTitanModel() {
		return c.processTitanStreamChunk(chunkData, ch)
	} else if c.isLlamaModel() {
		return c.processLlamaStreamChunk(chunkData, ch)
	}

	// Default to Claude
	return c.processClaudeStreamChunk(chunkData, ch)
}

// processClaudeStreamChunk processes Claude streaming chunks
func (c *Client) processClaudeStreamChunk(chunkData []byte, ch chan<- llm.StreamEvent) error {
	var chunk map[string]interface{}
	if err := json.Unmarshal(chunkData, &chunk); err != nil {
		return err
	}

	var text string
	if completion, ok := chunk["completion"].(string); ok {
		// Claude v2 format
		text = completion
	} else if delta, ok := chunk["delta"].(map[string]interface{}); ok {
		// Claude 3.x format
		if deltaText, ok := delta["text"].(string); ok {
			text = deltaText
		}
	}

	if text != "" {
		delta := &llm.MessageDelta{
			Content: []llm.MessageContent{llm.NewTextContent(text)},
		}
		ch <- llm.NewDeltaEvent(0, delta)
	}

	return nil
}

// processTitanStreamChunk processes Titan streaming chunks
func (c *Client) processTitanStreamChunk(chunkData []byte, ch chan<- llm.StreamEvent) error {
	var chunk map[string]interface{}
	if err := json.Unmarshal(chunkData, &chunk); err != nil {
		return err
	}

	if outputText, ok := chunk["outputText"].(string); ok {
		if outputText != "" {
			delta := &llm.MessageDelta{
				Content: []llm.MessageContent{llm.NewTextContent(outputText)},
			}
			ch <- llm.NewDeltaEvent(0, delta)
		}
	}

	return nil
}

// processLlamaStreamChunk processes Llama streaming chunks
func (c *Client) processLlamaStreamChunk(chunkData []byte, ch chan<- llm.StreamEvent) error {
	var chunk map[string]interface{}
	if err := json.Unmarshal(chunkData, &chunk); err != nil {
		return err
	}

	if generation, ok := chunk["generation"].(string); ok {
		if generation != "" {
			delta := &llm.MessageDelta{
				Content: []llm.MessageContent{llm.NewTextContent(generation)},
			}
			ch <- llm.NewDeltaEvent(0, delta)
		}
	}

	return nil
}

// Model type detection helpers
func (c *Client) isClaudeModel() bool {
	return strings.Contains(c.model, "claude") || strings.Contains(c.model, "anthropic")
}

func (c *Client) isTitanModel() bool {
	return strings.Contains(c.model, "titan") || strings.Contains(c.model, "amazon")
}

func (c *Client) isLlamaModel() bool {
	return strings.Contains(c.model, "llama") || strings.Contains(c.model, "meta")
}

// getMaxTokensForModel returns the maximum tokens for the given model
func (c *Client) getMaxTokensForModel(model string) int {
	// Claude models
	if strings.Contains(model, "claude-3") {
		return 200000 // Claude 3 has 200k context
	}
	if strings.Contains(model, "claude-v2") {
		return 100000 // Claude v2 has 100k context
	}

	// Titan models
	if strings.Contains(model, "titan") {
		return 8000 // Titan models typically have 8k context
	}

	// Llama models
	if strings.Contains(model, "llama") {
		if strings.Contains(model, "70b") {
			return 4096 // Llama 2 70B
		}
		return 2048 // Default for Llama models
	}

	// Default
	return 4000
}

// supportsTools checks if the model supports function calling
func (c *Client) supportsTools(model string) bool {
	// Claude 3 supports tools
	return strings.Contains(model, "claude-3")
}

// supportsVision checks if the model supports vision inputs
func (c *Client) supportsVision(model string) bool {
	// Claude 3 supports vision
	return strings.Contains(model, "claude-3")
}

// supportsFiles checks if the model supports file inputs
func (c *Client) supportsFiles(model string) bool {
	// Most Bedrock models can handle file content through context
	return true
}

// convertError converts errors to our internal error format
func (c *Client) convertError(err error) *llm.Error {
	if err == nil {
		return nil
	}

	// Check if it's already our error type
	if ourErr, ok := err.(*llm.Error); ok {
		return ourErr
	}

	errMsg := err.Error()

	// Check for authentication errors
	if strings.Contains(errMsg, "UnauthorizedOperation") ||
		strings.Contains(errMsg, "InvalidUserID.NotFound") ||
		strings.Contains(errMsg, "AuthFailure") {
		return &llm.Error{
			Code:       "authentication_error",
			Message:    errMsg,
			Type:       "authentication_error",
			StatusCode: 401,
		}
	}

	// Check for throttling errors
	if strings.Contains(errMsg, "ThrottlingException") ||
		strings.Contains(errMsg, "TooManyRequestsException") {
		return &llm.Error{
			Code:       "rate_limit_error",
			Message:    errMsg,
			Type:       "rate_limit_error",
			StatusCode: 429,
		}
	}

	// Check for model not found
	if strings.Contains(errMsg, "ValidationException") && strings.Contains(errMsg, "model") {
		return &llm.Error{
			Code:       "model_not_found",
			Message:    errMsg,
			Type:       "validation_error",
			StatusCode: 404,
		}
	}

	// Default error
	return &llm.Error{
		Code:    "api_error",
		Message: errMsg,
		Type:    "api_error",
	}
}
