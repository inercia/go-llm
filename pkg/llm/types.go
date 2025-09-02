// Core request and response types
package llm

// ChatRequest represents a chat completion request (provider-agnostic)
type ChatRequest struct {
	Model          string          `json:"model"`
	Messages       []Message       `json:"messages"`
	Tools          []Tool          `json:"tools,omitempty"`
	Temperature    *float32        `json:"temperature,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	TopP           *float32        `json:"top_p,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ChatResponse represents a chat completion response (provider-agnostic)
type ChatResponse struct {
	ID      string   `json:"id"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage,omitempty"`
}

// Choice represents a single response choice
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// Usage represents token usage information
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// WantsToolExecution checks if this choice indicates the LLM wants to execute tools
func (c Choice) WantsToolExecution() bool {
	return c.FinishReason == FinishReasonToolCalls || c.Message.HasToolCalls()
}

// IsComplete checks if this choice represents a complete response (not requiring tool execution)
func (c Choice) IsComplete() bool {
	return c.FinishReason == FinishReasonStop || c.FinishReason == FinishReasonLength
}

// RequiresToolExecution checks if this response requires tool execution before continuing
func (r ChatResponse) RequiresToolExecution() bool {
	for _, choice := range r.Choices {
		if choice.WantsToolExecution() {
			return true
		}
	}
	return false
}

// GetToolCalls returns all tool calls from all choices in the response
func (r ChatResponse) GetToolCalls() []ToolCall {
	var allToolCalls []ToolCall
	for _, choice := range r.Choices {
		allToolCalls = append(allToolCalls, choice.Message.ToolCalls...)
	}
	return allToolCalls
}

// DeepCopy creates a deep copy of the ChatResponse, including all choices and usage information
// This method ensures that modifications to the copy will not affect the original response,
// preventing issues with shared mutable state in concurrent tracing environments
func (r ChatResponse) DeepCopy() ChatResponse {
	copy := ChatResponse{
		ID:    r.ID,
		Model: r.Model,
		Usage: Usage{
			PromptTokens:     r.Usage.PromptTokens,
			CompletionTokens: r.Usage.CompletionTokens,
			TotalTokens:      r.Usage.TotalTokens,
		},
	}

	// Deep copy the Choices slice
	if len(r.Choices) > 0 {
		copy.Choices = make([]Choice, 0, len(r.Choices))
		for _, choice := range r.Choices {
			copy.Choices = append(copy.Choices, Choice{
				Index:        choice.Index,
				Message:      choice.Message.DeepCopy(), // Use Message's DeepCopy method
				FinishReason: choice.FinishReason,
			})
		}
	}

	return copy
}
