package mock

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/inercia/go-llm/pkg/llm"
)

// secureRandomFloat64 generates a cryptographically secure random float64 between 0 and 1
func secureRandomFloat64() (float64, error) {
	var bytes [8]byte
	_, err := rand.Read(bytes[:])
	if err != nil {
		return 0, err
	}
	// Convert bytes to uint64, then to float64 between 0 and 1
	return float64(binary.BigEndian.Uint64(bytes[:])) / float64(^uint64(0)), nil
}

// Client implements the llm.Client interface for testing
type Client struct {
	modelInfo         llm.ModelInfo
	responses         []llm.ChatResponse
	responseIndex     int
	errors            []error
	errorIndex        int
	callLog           []llm.ChatRequest
	streamResponses   [][]llm.StreamEvent
	streamIndex       int
	latencySimulation time.Duration
	failureRate       float64
	conversationState map[string]interface{}
	toolCallHandlers  map[string]func(args string) (string, error)

	// Health check caching (even for mock)
	lastHealthCheck  *time.Time
	lastHealthStatus *bool
}

// NewClient creates a new mock LLM client for testing
func NewClient(modelName, provider string) (*Client, error) {
	return &Client{
		modelInfo: llm.ModelInfo{
			Name:              modelName,
			Provider:          provider,
			MaxTokens:         4096,
			SupportsTools:     true,
			SupportsVision:    false,
			SupportsFiles:     false,
			SupportsStreaming: true,
		},
		responses:         []llm.ChatResponse{},
		responseIndex:     0,
		errors:            []error{},
		errorIndex:        0,
		callLog:           []llm.ChatRequest{},
		streamResponses:   [][]llm.StreamEvent{},
		streamIndex:       0,
		latencySimulation: 0,
		failureRate:       0,
		conversationState: make(map[string]interface{}),
		toolCallHandlers:  make(map[string]func(args string) (string, error)),
	}, nil
}

// handleToolResponse generates a response after a tool call
func (m *Client) handleToolResponse(req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Find the last tool message
	var toolResult string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == llm.RoleTool {
			if len(req.Messages[i].Content) > 0 {
				if textContent, ok := req.Messages[i].Content[0].(*llm.TextContent); ok {
					toolResult = textContent.Text
				}
			}
			break
		}
	}

	response := fmt.Sprintf("Based on the tool result: %s, I can provide you with the following information...", toolResult)

	return &llm.ChatResponse{
		ID:    fmt.Sprintf("mock-tool-resp-%d", time.Now().UnixNano()),
		Model: req.Model,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.MessageContent{llm.NewTextContent(response)},
				},
				FinishReason: "stop",
			},
		},
		Usage: llm.Usage{
			PromptTokens:     len(strings.Split(toolResult, " ")) + 10,
			CompletionTokens: len(strings.Split(response, " ")),
			TotalTokens:      len(strings.Split(toolResult, " ")) + len(strings.Split(response, " ")) + 10,
		},
	}, nil
}

// generateIntelligentResponse creates context-aware responses
func (m *Client) generateIntelligentResponse(req llm.ChatRequest) (*llm.ChatResponse, error) {
	var lastUserMessage string
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == llm.RoleUser && len(req.Messages[i].Content) > 0 {
			if textContent, ok := req.Messages[i].Content[0].(*llm.TextContent); ok {
				lastUserMessage = textContent.Text
			}
			break
		}
	}

	// Analyze the message for tool calling opportunities
	lowerMsg := strings.ToLower(lastUserMessage)

	// Check if this looks like a request that would need tools
	toolTriggers := []string{"search", "calculate", "weather", "email", "file", "database", "api"}
	for _, trigger := range toolTriggers {
		if strings.Contains(lowerMsg, trigger) {
			return m.generateToolCallResponse(req, trigger, lastUserMessage)
		}
	}

	// Generate appropriate text response
	var response string
	switch {
	case strings.Contains(lowerMsg, "hello") || strings.Contains(lowerMsg, "hi"):
		response = "Hello! How can I help you today?"
	case strings.Contains(lowerMsg, "help"):
		response = "I'm here to help! I can assist with various tasks including searching, calculations, and more."
	case strings.Contains(lowerMsg, "test"):
		response = "This is a mock response for testing purposes. The system is working correctly."
	default:
		response = fmt.Sprintf("I understand you're asking about: %s. Let me help you with that.", lastUserMessage)
	}

	return &llm.ChatResponse{
		ID:    fmt.Sprintf("mock-resp-%d", time.Now().UnixNano()),
		Model: req.Model,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.MessageContent{llm.NewTextContent(response)},
				},
				FinishReason: "stop",
			},
		},
		Usage: llm.Usage{
			PromptTokens:     len(strings.Split(lastUserMessage, " ")) + 5,
			CompletionTokens: len(strings.Split(response, " ")),
			TotalTokens:      len(strings.Split(lastUserMessage, " ")) + len(strings.Split(response, " ")) + 5,
		},
	}, nil
}

// generateToolCallResponse creates a response with appropriate tool calls
func (m *Client) generateToolCallResponse(req llm.ChatRequest, trigger, userMessage string) (*llm.ChatResponse, error) {
	var toolCall llm.ToolCall

	switch trigger {
	case "search":
		args, _ := json.Marshal(map[string]string{"query": userMessage})
		toolCall = llm.ToolCall{
			ID:   fmt.Sprintf("call-search-%d", time.Now().UnixNano()),
			Type: "function",
			Function: llm.ToolCallFunction{
				Name:      "web_search",
				Arguments: string(args),
			},
		}
	case "calculate":
		args, _ := json.Marshal(map[string]string{"expression": userMessage})
		toolCall = llm.ToolCall{
			ID:   fmt.Sprintf("call-calc-%d", time.Now().UnixNano()),
			Type: "function",
			Function: llm.ToolCallFunction{
				Name:      "calculator",
				Arguments: string(args),
			},
		}
	case "weather":
		args, _ := json.Marshal(map[string]string{"location": "user_location"})
		toolCall = llm.ToolCall{
			ID:   fmt.Sprintf("call-weather-%d", time.Now().UnixNano()),
			Type: "function",
			Function: llm.ToolCallFunction{
				Name:      "get_weather",
				Arguments: string(args),
			},
		}
	default:
		args, _ := json.Marshal(map[string]string{"request": userMessage})
		toolCall = llm.ToolCall{
			ID:   fmt.Sprintf("call-generic-%d", time.Now().UnixNano()),
			Type: "function",
			Function: llm.ToolCallFunction{
				Name:      fmt.Sprintf("%s_tool", trigger),
				Arguments: string(args),
			},
		}
	}

	return &llm.ChatResponse{
		ID:    fmt.Sprintf("mock-tool-call-%d", time.Now().UnixNano()),
		Model: req.Model,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:      llm.RoleAssistant,
					Content:   []llm.MessageContent{llm.NewTextContent("I need to use a tool to help with your request.")},
					ToolCalls: []llm.ToolCall{toolCall},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: llm.Usage{
			PromptTokens:     len(strings.Split(userMessage, " ")) + 10,
			CompletionTokens: 15,
			TotalTokens:      len(strings.Split(userMessage, " ")) + 25,
		},
	}, nil
}

// ChatCompletion returns pre-configured responses or errors
func (m *Client) ChatCompletion(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	// Log the request for testing assertions
	m.callLog = append(m.callLog, req)

	// Simulate latency if configured
	if m.latencySimulation > 0 {
		select {
		case <-time.After(m.latencySimulation):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Simulate random failures if configured
	if m.failureRate > 0 {
		randomValue, err := secureRandomFloat64()
		if err != nil {
			// If we can't generate secure random, fall back to no failure simulation
			randomValue = 0
		}
		if randomValue < m.failureRate {
			return nil, &llm.Error{
				Code:    "mock_random_failure",
				Message: "Simulated random failure",
				Type:    "simulation_error",
			}
		}
	}

	// Check for tool calls in the request and handle them
	if len(req.Messages) > 0 {
		lastMsg := req.Messages[len(req.Messages)-1]
		if lastMsg.Role == llm.RoleTool {
			// This is a tool response, generate appropriate follow-up
			return m.handleToolResponse(req)
		}
	}

	// Return error if configured
	if m.errorIndex < len(m.errors) {
		err := m.errors[m.errorIndex]
		m.errorIndex++
		return nil, err
	}

	// Return response if configured
	if m.responseIndex < len(m.responses) {
		resp := m.responses[m.responseIndex]
		m.responseIndex++
		return &resp, nil
	}

	// Generate intelligent response based on message content
	return m.generateIntelligentResponse(req)
}

// StreamChatCompletion simulates streaming by sending chunked events
func (m *Client) StreamChatCompletion(ctx context.Context, req llm.ChatRequest) (<-chan llm.StreamEvent, error) {
	// Log the stream request
	m.callLog = append(m.callLog, req)

	// Simulate latency if configured
	if m.latencySimulation > 0 {
		select {
		case <-time.After(m.latencySimulation):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Return error if configured for first call
	if m.errorIndex < len(m.errors) {
		err := m.errors[m.errorIndex]
		m.errorIndex++
		ch := make(chan llm.StreamEvent, 1)
		ch <- llm.NewErrorEvent(&llm.Error{
			Code:    "mock_error",
			Message: err.Error(),
			Type:    "simulation_error",
		})
		close(ch)
		return ch, nil
	}

	// Return pre-configured stream if available
	if m.streamIndex < len(m.streamResponses) {
		events := m.streamResponses[m.streamIndex]
		m.streamIndex++
		return m.sendStreamEvents(ctx, events), nil
	}

	// Generate intelligent streaming response
	return m.generateStreamingResponse(ctx, req), nil
}

// sendStreamEvents sends pre-configured stream events
func (m *Client) sendStreamEvents(ctx context.Context, events []llm.StreamEvent) <-chan llm.StreamEvent {
	ch := make(chan llm.StreamEvent, len(events))

	go func() {
		defer close(ch)
		for _, event := range events {
			select {
			case <-ctx.Done():
				return
			case ch <- event:
			}
			// Simulate streaming delay
			time.Sleep(50 * time.Millisecond)
		}
	}()

	return ch
}

// generateStreamingResponse creates intelligent streaming responses
func (m *Client) generateStreamingResponse(ctx context.Context, req llm.ChatRequest) <-chan llm.StreamEvent {
	ch := make(chan llm.StreamEvent, 20)

	go func() {
		defer close(ch)

		// Determine response type based on request
		var fullText string
		var shouldCallTool bool
		var toolName string

		// Extract user message for analysis
		var userMessage string
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == llm.RoleUser && len(req.Messages[i].Content) > 0 {
				if textContent, ok := req.Messages[i].Content[0].(*llm.TextContent); ok {
					userMessage = textContent.Text
				}
				break
			}
		}

		lowerMsg := strings.ToLower(userMessage)

		// Check if we should call a tool
		if strings.Contains(lowerMsg, "search") {
			shouldCallTool = true
			toolName = "web_search"
			fullText = "I'll search for that information for you."
		} else if strings.Contains(lowerMsg, "calculate") {
			shouldCallTool = true
			toolName = "calculator"
			fullText = "Let me calculate that for you."
		} else {
			fullText = "This is a streamed mock response that demonstrates chunked delivery of content for testing purposes."
		}

		// Stream the text response
		words := strings.Split(fullText, " ")
		for _, word := range words {
			select {
			case <-ctx.Done():
				return
			case ch <- llm.NewDeltaEvent(0, &llm.MessageDelta{
				Content: []llm.MessageContent{llm.NewTextContent(word + " ")},
			}):
			}
			time.Sleep(100 * time.Millisecond)
		}

		// Add tool call if needed
		if shouldCallTool {
			select {
			case <-ctx.Done():
				return
			case ch <- llm.NewDeltaEvent(0, &llm.MessageDelta{
				ToolCalls: []llm.ToolCallDelta{
					{
						Index: 0,
						ID:    fmt.Sprintf("call-%s-%d", toolName, time.Now().UnixNano()),
						Type:  "function",
						Function: &llm.ToolCallFunctionDelta{
							Name:      toolName,
							Arguments: fmt.Sprintf(`{"query": "%s"}`, userMessage),
						},
					},
				},
			}):
			}

			select {
			case <-ctx.Done():
				return
			case ch <- llm.NewDoneEvent(0, "tool_calls"):
			}
		} else {
			select {
			case <-ctx.Done():
				return
			case ch <- llm.NewDoneEvent(0, "stop"):
			}
		}
	}()

	return ch
}

// GetRemote returns information about the remote client
func (m *Client) GetRemote() llm.ClientRemoteInfo {
	info := llm.ClientRemoteInfo{
		Name: "mock",
	}

	// Check if we need to refresh the health status
	now := time.Now()
	needsRefresh := m.lastHealthCheck == nil ||
		now.Sub(*m.lastHealthCheck) >= llm.DefaultHealthCheckInterval

	if needsRefresh {
		healthy := true // Mock client is always healthy
		m.lastHealthStatus = &healthy
		m.lastHealthCheck = &now
	}

	info.Status = &llm.ClientRemoteInfoStatus{
		Healthy:     m.lastHealthStatus,
		LastChecked: m.lastHealthCheck,
	}

	return info
}

// GetModelInfo returns the configured model info
func (m *Client) GetModelInfo() llm.ModelInfo {
	return m.modelInfo
}

// Close does nothing for mock client
func (m *Client) Close() error {
	return nil
}

// StreamChatCompletionWithTools performs streaming chat completion with tool execution capabilities
func (m *Client) StreamChatCompletionWithTools(ctx context.Context, req llm.ChatRequest, toolStream <-chan llm.StreamEvent) (<-chan llm.StreamEvent, error) {
	// Log the request
	m.callLog = append(m.callLog, req)

	// Simulate latency if configured
	if m.latencySimulation > 0 {
		select {
		case <-time.After(m.latencySimulation):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Get LLM stream
	llmStream, err := m.StreamChatCompletion(ctx, req)
	if err != nil {
		return nil, err
	}

	// Prepare tool streams
	var toolStreams []<-chan llm.StreamEvent
	if toolStream != nil {
		toolStreams = append(toolStreams, toolStream)
	}

	// Create stream merger
	merger := llm.NewStreamMerger(ctx, llmStream, toolStreams)

	// Start merging and return the output channel
	return merger.Start(), nil
}

// Test helper methods

// AddResponse adds a response to be returned by subsequent calls
func (m *Client) AddResponse(response llm.ChatResponse) *Client {
	m.responses = append(m.responses, response)
	return m
}

// AddError adds an error to be returned by subsequent calls
func (m *Client) AddError(err error) *Client {
	m.errors = append(m.errors, err)
	return m
}

// GetCallLog returns all requests made to this mock client
func (m *Client) GetCallLog() []llm.ChatRequest {
	return m.callLog
}

// GetLastCall returns the most recent request made to this mock client
func (m *Client) GetLastCall() *llm.ChatRequest {
	if len(m.callLog) == 0 {
		return nil
	}
	return &m.callLog[len(m.callLog)-1]
}

// Reset clears all responses, errors, and call logs
func (m *Client) Reset() *Client {
	m.responses = []llm.ChatResponse{}
	m.responseIndex = 0
	m.errors = []error{}
	m.errorIndex = 0
	m.callLog = []llm.ChatRequest{}
	return m
}

// Convenience methods for common test scenarios

// WithSimpleResponse adds a simple text response
func (m *Client) WithSimpleResponse(content string) *Client {
	return m.AddResponse(llm.ChatResponse{
		ID:    fmt.Sprintf("mock-simple-%d", time.Now().UnixNano()),
		Model: m.modelInfo.Name,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.MessageContent{llm.NewTextContent(content)},
				},
				FinishReason: "stop",
			},
		},
	})
}

// WithToolCall adds a response that includes a tool call
func (m *Client) WithToolCall(toolName string, args map[string]interface{}) *Client {
	argsJSON := "{}"
	if len(args) > 0 {
		// In a real implementation, we'd properly serialize this
		argsJSON = `{"mock_args": true}`
	}

	return m.AddResponse(llm.ChatResponse{
		ID:    fmt.Sprintf("mock-tool-%d", time.Now().UnixNano()),
		Model: m.modelInfo.Name,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: []llm.MessageContent{llm.NewTextContent("I need to use a tool to help with this request.")},
					ToolCalls: []llm.ToolCall{
						{
							ID:   fmt.Sprintf("call-%d", time.Now().UnixNano()),
							Type: "function",
							Function: llm.ToolCallFunction{
								Name:      toolName,
								Arguments: argsJSON,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	})
}

// WithError adds an error response
func (m *Client) WithError(code, message, errorType string) *Client {
	return m.AddError(&llm.Error{
		Code:    code,
		Message: message,
		Type:    errorType,
	})
}

// Configuration methods for enhanced testing

// WithLatency configures simulated latency for requests
func (m *Client) WithLatency(duration time.Duration) *Client {
	m.latencySimulation = duration
	return m
}

// WithFailureRate configures random failure simulation (0.0 to 1.0)
func (m *Client) WithFailureRate(rate float64) *Client {
	m.failureRate = rate
	return m
}

// WithModelCapabilities configures the model's capabilities
func (m *Client) WithModelCapabilities(maxTokens int, supportsTools, supportsVision, supportsFiles, supportsStreaming bool) *Client {
	m.modelInfo.MaxTokens = maxTokens
	m.modelInfo.SupportsTools = supportsTools
	m.modelInfo.SupportsVision = supportsVision
	m.modelInfo.SupportsFiles = supportsFiles
	m.modelInfo.SupportsStreaming = supportsStreaming
	return m
}

// WithConversationState sets conversation state for context-aware responses
func (m *Client) WithConversationState(key string, value interface{}) *Client {
	m.conversationState[key] = value
	return m
}

// WithStreamResponse adds a pre-configured streaming response
func (m *Client) WithStreamResponse(events []llm.StreamEvent) *Client {
	m.streamResponses = append(m.streamResponses, events)
	return m
}

// WithToolCallHandler registers a handler for specific tool calls
func (m *Client) WithToolCallHandler(toolName string, handler func(args string) (string, error)) *Client {
	m.toolCallHandlers[toolName] = handler
	return m
}

// Multi-turn conversation helpers

// WithConversation sets up a multi-turn conversation scenario
func (m *Client) WithConversation(exchanges []ConversationExchange) *Client {
	for _, exchange := range exchanges {
		if exchange.ToolCall != nil {
			m.WithToolCall(exchange.ToolCall.Name, exchange.ToolCall.Arguments)
		}
		if exchange.Response != "" {
			m.WithSimpleResponse(exchange.Response)
		}
	}
	return m
}

// ConversationExchange represents a turn in a conversation
type ConversationExchange struct {
	Response string
	ToolCall *MockToolCall
}

// MockToolCall represents a tool call for testing
type MockToolCall struct {
	Name      string
	Arguments map[string]interface{}
}

// Advanced response builders

// WithSystemMessage creates a response as if from a system message
func (m *Client) WithSystemMessage(content string) *Client {
	return m.AddResponse(llm.ChatResponse{
		ID:    fmt.Sprintf("mock-system-%d", time.Now().UnixNano()),
		Model: m.modelInfo.Name,
		Choices: []llm.Choice{
			{
				Index: 0,
				Message: llm.Message{
					Role:    llm.RoleSystem,
					Content: []llm.MessageContent{llm.NewTextContent(content)},
				},
				FinishReason: "stop",
			},
		},
	})
}

// WithFunctionResult creates a response that follows a function call
func (m *Client) WithFunctionResult(functionName, result string) *Client {
	response := fmt.Sprintf("Based on the %s function result: %s", functionName, result)
	return m.WithSimpleResponse(response)
}

// WithMultiStepResponse creates a complex response with reasoning steps
func (m *Client) WithMultiStepResponse(steps []string) *Client {
	content := "Let me work through this step by step:\n\n"
	for i, step := range steps {
		content += fmt.Sprintf("%d. %s\n", i+1, step)
	}
	content += "\nBased on this analysis, here's my conclusion..."
	return m.WithSimpleResponse(content)
}

// Streaming response helpers

// CreateWordByWordStream creates a streaming response that sends words individually
func CreateWordByWordStream(text string, delay time.Duration) []llm.StreamEvent {
	words := strings.Split(text, " ")
	events := make([]llm.StreamEvent, 0, len(words)+1)

	for _, word := range words {
		events = append(events, llm.NewDeltaEvent(0, &llm.MessageDelta{
			Content: []llm.MessageContent{llm.NewTextContent(word + " ")},
		}))
	}

	events = append(events, llm.NewDoneEvent(0, "stop"))
	return events
}

// CreateToolCallStream creates a streaming response that includes a tool call
func CreateToolCallStream(initialText, toolName string, args map[string]interface{}) []llm.StreamEvent {
	events := make([]llm.StreamEvent, 0, 5)

	// Initial response text
	if initialText != "" {
		words := strings.Split(initialText, " ")
		for _, word := range words {
			events = append(events, llm.NewDeltaEvent(0, &llm.MessageDelta{
				Content: []llm.MessageContent{llm.NewTextContent(word + " ")},
			}))
		}
	}

	// Tool call
	argsJSON, _ := json.Marshal(args)
	events = append(events, llm.NewDeltaEvent(0, &llm.MessageDelta{
		ToolCalls: []llm.ToolCallDelta{
			{
				Index: 0,
				ID:    fmt.Sprintf("call-%s-%d", toolName, time.Now().UnixNano()),
				Type:  "function",
				Function: &llm.ToolCallFunctionDelta{
					Name:      toolName,
					Arguments: string(argsJSON),
				},
			},
		},
	}))

	events = append(events, llm.NewDoneEvent(0, "tool_calls"))
	return events
}

// Test assertion helpers

// AssertCallCount verifies the number of calls made
func (m *Client) AssertCallCount(expected int) bool {
	return len(m.callLog) == expected
}

// AssertLastMessageContains checks if the last user message contains specific text
func (m *Client) AssertLastMessageContains(text string) bool {
	lastCall := m.GetLastCall()
	if lastCall == nil {
		return false
	}

	for _, msg := range lastCall.Messages {
		if msg.Role == llm.RoleUser {
			for _, content := range msg.Content {
				if textContent, ok := content.(*llm.TextContent); ok {
					if strings.Contains(textContent.Text, text) {
						return true
					}
				}
			}
		}
	}
	return false
}

// AssertToolWasCalled checks if a specific tool was called
func (m *Client) AssertToolWasCalled(toolName string) bool {
	for _, call := range m.callLog {
		for _, msg := range call.Messages {
			if msg.Role == llm.RoleAssistant {
				for _, toolCall := range msg.ToolCalls {
					if toolCall.Function.Name == toolName {
						return true
					}
				}
			}
		}
	}
	return false
}

// Example test usage scenarios:

/*
// Basic Testing Example
func TestBasicLLMInteraction(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithSimpleResponse("Hello! How can I help you?")

	response, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Hello")}},
		},
	})

	assert.NoError(t, err)
	assert.Equal(t, "Hello! How can I help you?", response.Choices[0].Message.Content[0].(*TextContent).Text)
}

// Tool Calling Example
func TestToolCallingScenario(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithToolCall("web_search", map[string]interface{}{"query": "Go programming"})
		.WithFunctionResult("web_search", "Go is a programming language developed by Google")

	// First call - tool call
	resp1, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Search for Go programming info")}},
		},
	})
	assert.NoError(t, err)
	assert.Equal(t, "tool_calls", resp1.Choices[0].FinishReason)

	// Second call - after tool result
	resp2, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Search for Go programming info")}},
			resp1.Choices[0].Message,
			{Role: RoleTool, Content: []MessageContent{NewTextContent("Go is a programming language developed by Google")}},
		},
	})
	assert.NoError(t, err)
	assert.Contains(t, resp2.Choices[0].Message.Content[0].(*TextContent).Text, "Google")
}

// Multi-turn Conversation Example
func TestMultiTurnConversation(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithConversation([]ConversationExchange{
			{Response: "I'd be happy to help you learn Go!"},
			{ToolCall: &MockToolCall{Name: "get_documentation", Arguments: map[string]interface{}{"topic": "golang basics"}}},
			{Response: "Based on the documentation, here are the Go basics..."},
		})

	// Simulate conversation turns
	responses := make([]*ChatResponse, 3)
	messages := []Message{
		{Role: RoleUser, Content: []MessageContent{NewTextContent("Help me learn Go")}},
	}

	for i := 0; i < 3; i++ {
		resp, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
			Model: "gpt-4",
			Messages: messages,
		})
		assert.NoError(t, err)
		responses[i] = resp
		messages = append(messages, resp.Choices[0].Message)
	}

	assert.True(t, mockLLM.AssertCallCount(3))
}

// Streaming Example
func TestStreamingResponse(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithStreamResponse(CreateWordByWordStream("This is a streaming response", 10*time.Millisecond))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := mockLLM.StreamChatCompletion(ctx, ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Tell me something")}},
		},
	})
	assert.NoError(t, err)

	var fullText strings.Builder
	for event := range stream {
		if event.IsDelta() && event.Choice.Delta.Content != nil {
			if textContent, ok := event.Choice.Delta.Content[0].(*TextContent); ok {
				fullText.WriteString(textContent.Text)
			}
		}
	}

	assert.Equal(t, "This is a streaming response ", fullText.String())
}

// Error Handling Example
func TestErrorHandling(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithError("rate_limit", "Rate limit exceeded", "api_error")
		.WithSimpleResponse("Recovery response after error")

	// First call should error
	_, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Hello")}},
		},
	})
	assert.Error(t, err)

	// Second call should succeed
	resp, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Hello again")}},
		},
	})
	assert.NoError(t, err)
	assert.Equal(t, "Recovery response after error", resp.Choices[0].Message.Content[0].(*TextContent).Text)
}

// Advanced Configuration Example
func TestAdvancedConfiguration(t *testing.T) {
	mockLLM := NewMockClient("custom-model", "mock")
		.WithLatency(100 * time.Millisecond)
		.WithFailureRate(0.1) // 10% failure rate
		.WithModelCapabilities(8192, true, true, false, true)
		.WithConversationState("user_preference", "detailed_explanations")

	start := time.Now()
	_, err := mockLLM.ChatCompletion(context.Background(), ChatRequest{
		Model: "custom-model",
		Messages: []Message{
			{Role: RoleUser, Content: []MessageContent{NewTextContent("Hello")}},
		},
	})

	if err == nil {
		// If no random failure occurred, check latency was simulated
		elapsed := time.Since(start)
		assert.True(t, elapsed >= 100*time.Millisecond)
	}

	modelInfo := mockLLM.GetModelInfo()
	assert.Equal(t, 8192, modelInfo.MaxTokens)
	assert.True(t, modelInfo.SupportsVision)
}

// Custom Tool Handler Example
func TestCustomToolHandler(t *testing.T) {
	calculator := func(args string) (string, error) {
		// Simple calculator implementation for testing
		return "42", nil
	}

	mockLLM := NewMockClient("gpt-4", "mock")
		.WithToolCallHandler("calculator", calculator)

	// This would use the custom handler in a real implementation
	// For now, we just verify it's registered
	assert.NotNil(t, mockLLM.toolCallHandlers["calculator"])
}

// Agent Testing Example
func TestAgentWithMockLLM(t *testing.T) {
	mockLLM := NewMockClient("gpt-4", "mock")
		.WithSimpleResponse("I'll help you search for that information.")
		.WithToolCall("search_web", map[string]interface{}{"query": "Go programming best practices"})
		.WithSimpleResponse("Based on my search, here are the key Go programming best practices...")

	// Assuming you have an Agent type that uses the LLM
	// agent := NewAgentWithLLM(mockLLM, logger, tools)
	// result, err := agent.ProcessRequest("What are Go programming best practices?")

	// Verify the LLM interactions
	assert.True(t, mockLLM.AssertCallCount(3))
	assert.True(t, mockLLM.AssertLastMessageContains("Go programming"))
	assert.True(t, mockLLM.AssertToolWasCalled("search_web"))
}
*/
