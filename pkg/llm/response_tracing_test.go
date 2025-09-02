package llm

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TracingMiddleware simulates the tracing middleware described in the issue
type TracingMiddleware struct {
	name              string
	capturedResponses []*ChatResponse
	capturedContent   []map[string]interface{}
}

func NewTracingMiddleware(name string) *TracingMiddleware {
	return &TracingMiddleware{
		name:              name,
		capturedResponses: make([]*ChatResponse, 0),
		capturedContent:   make([]map[string]interface{}, 0),
	}
}

func (t *TracingMiddleware) Name() string {
	return t.name
}

func (t *TracingMiddleware) ProcessRequest(ctx context.Context, req *ChatRequest) (*ChatRequest, error) {
	return req, nil
}

func (t *TracingMiddleware) ProcessResponse(ctx context.Context, req *ChatRequest, resp *ChatResponse, err error) (*ChatResponse, error) {
	if resp != nil && err == nil {
		// Capture the response for analysis
		t.capturedResponses = append(t.capturedResponses, resp)

		// Simulate the extractLLMResponseContent function
		content := t.extractLLMResponseContent(resp)
		t.capturedContent = append(t.capturedContent, content)
	}
	return resp, err
}

func (t *TracingMiddleware) ProcessStreamEvent(ctx context.Context, req *ChatRequest, event StreamEvent) (StreamEvent, error) {
	return event, nil
}

// extractLLMResponseContent simulates the function mentioned in the issue
func (t *TracingMiddleware) extractLLMResponseContent(response *ChatResponse) map[string]interface{} {
	if response == nil {
		return map[string]interface{}{}
	}

	content := map[string]interface{}{
		"response_id":       response.ID,
		"model":             response.Model,
		"choice_count":      len(response.Choices),
		"total_tokens":      0,
		"completion_tokens": 0,
	}

	if response.Usage.TotalTokens > 0 {
		content["total_tokens"] = response.Usage.TotalTokens
		content["completion_tokens"] = response.Usage.CompletionTokens
	}

	if len(response.Choices) > 0 {
		choices := make([]map[string]interface{}, 0, len(response.Choices))
		for _, choice := range response.Choices {
			choiceContent := map[string]interface{}{
				"index":         choice.Index,
				"finish_reason": choice.FinishReason,
				"content_size":  0,
				"text_preview":  "",
			}

			// Extract text content
			text := choice.Message.GetText()
			if text != "" {
				choiceContent["text_preview"] = text
				choiceContent["content_size"] = len(text)
			}

			choices = append(choices, choiceContent)
		}
		content["choices"] = choices
	}

	return content
}

// GetLastCapturedResponse returns the last captured response for testing
func (t *TracingMiddleware) GetLastCapturedResponse() *ChatResponse {
	if len(t.capturedResponses) == 0 {
		return nil
	}
	return t.capturedResponses[len(t.capturedResponses)-1]
}

// GetLastCapturedContent returns the last captured content for testing
func (t *TracingMiddleware) GetLastCapturedContent() map[string]interface{} {
	if len(t.capturedContent) == 0 {
		return nil
	}
	return t.capturedContent[len(t.capturedContent)-1]
}

// TestResponseTracingScenario reproduces the scenario described in the issue
func TestResponseTracingScenario(t *testing.T) {
	// Create a mock response that simulates a successful LLM response
	originalResponse := &ChatResponse{
		ID:    "chatcmpl-14rsdzsqecyiztgjdni6tp",
		Model: "openai/gpt-oss-20b",
		Choices: []Choice{
			{
				Index:        0,
				Message:      NewTextMessage(RoleAssistant, "Hello World"),
				FinishReason: "stop",
			},
		},
		Usage: Usage{
			PromptTokens:     76,
			CompletionTokens: 17,
			TotalTokens:      93,
		},
	}

	t.Run("direct_response_processing", func(t *testing.T) {
		// Test direct processing without middleware - should work
		tracer := NewTracingMiddleware("test-tracer")

		ctx := context.Background()
		req := &ChatRequest{
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello"),
			},
		}

		// Simulate direct processing
		processedResp, err := tracer.ProcessResponse(ctx, req, originalResponse, nil)

		require.NoError(t, err)
		require.NotNil(t, processedResp)

		// Verify the response is still complete
		assert.Equal(t, originalResponse.ID, processedResp.ID)
		assert.Len(t, processedResp.Choices, 1)
		assert.Equal(t, "Hello World", processedResp.Choices[0].Message.GetText())

		// Check captured content
		capturedContent := tracer.GetLastCapturedContent()
		require.NotNil(t, capturedContent)

		t.Logf("Captured content: %+v", capturedContent)

		// Verify the content extraction worked correctly
		assert.Equal(t, "chatcmpl-14rsdzsqecyiztgjdni6tp", capturedContent["response_id"])
		assert.Equal(t, 1, capturedContent["choice_count"])
		assert.Equal(t, 93, capturedContent["total_tokens"])

		// Check choices array
		choices, ok := capturedContent["choices"].([]map[string]interface{})
		require.True(t, ok, "choices should be an array")
		require.Len(t, choices, 1)

		choice := choices[0]
		assert.Equal(t, "Hello World", choice["text_preview"])
		assert.Equal(t, 11, choice["content_size"]) // len("Hello World")
		assert.Equal(t, "stop", choice["finish_reason"])
	})

	t.Run("middleware_chain_scenario", func(t *testing.T) {
		// Test with middleware chain - simulates the real scenario
		tracer := NewTracingMiddleware("trace-middleware")

		// Create enhanced client with tracing middleware
		mockClient := NewMockClient("openai/gpt-oss-20b", "openai")
		enhancedClient := NewEnhancedClient(mockClient, []Middleware{tracer})

		// Configure mock to return our test response
		mockClient.AddResponse(*originalResponse)

		ctx := context.Background()
		req := ChatRequest{
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello"),
			},
		}

		// Execute through middleware chain
		finalResp, err := enhancedClient.ChatCompletion(ctx, req)

		require.NoError(t, err)
		require.NotNil(t, finalResp)

		// Verify final response is still complete
		assert.Equal(t, originalResponse.ID, finalResp.ID)
		assert.Len(t, finalResp.Choices, 1)
		assert.Equal(t, "Hello World", finalResp.Choices[0].Message.GetText())

		// Check what the tracer captured
		capturedResponse := tracer.GetLastCapturedResponse()
		capturedContent := tracer.GetLastCapturedContent()

		require.NotNil(t, capturedResponse, "Tracer should have captured the response")
		require.NotNil(t, capturedContent, "Tracer should have captured content")

		t.Logf("Final response text: %s", finalResp.Choices[0].Message.GetText())
		t.Logf("Captured response text: %s", capturedResponse.Choices[0].Message.GetText())
		t.Logf("Captured content: %+v", capturedContent)

		// Verify the captured response has content
		assert.Equal(t, "Hello World", capturedResponse.Choices[0].Message.GetText())

		// Verify the extracted content shows the text
		choices, ok := capturedContent["choices"].([]map[string]interface{})
		require.True(t, ok, "choices should be an array")
		require.Len(t, choices, 1)

		choice := choices[0]
		if choice["text_preview"] == "" || choice["content_size"] == 0 {
			t.Errorf("ISSUE REPRODUCED: Empty content in trace - text_preview='%v', content_size=%v",
				choice["text_preview"], choice["content_size"])

			// Debug: Check the response structure
			respJSON, _ := json.MarshalIndent(capturedResponse, "", "  ")
			t.Logf("Captured response JSON:\n%s", string(respJSON))

			contentJSON, _ := json.MarshalIndent(capturedContent, "", "  ")
			t.Logf("Captured content JSON:\n%s", string(contentJSON))
		} else {
			assert.Equal(t, "Hello World", choice["text_preview"])
			assert.Equal(t, 11, choice["content_size"])
		}
	})

	t.Run("json_serialization_round_trip", func(t *testing.T) {
		// Test if the issue is related to JSON serialization/deserialization

		// Serialize the original response
		originalJSON, err := json.Marshal(originalResponse)
		require.NoError(t, err)

		t.Logf("Original response JSON: %s", string(originalJSON))

		// Deserialize it back
		var deserializedResponse ChatResponse
		err = json.Unmarshal(originalJSON, &deserializedResponse)
		require.NoError(t, err)

		// Check if content survives round-trip
		require.Len(t, deserializedResponse.Choices, 1)

		deserializedText := deserializedResponse.Choices[0].Message.GetText()
		t.Logf("Deserialized response text: '%s'", deserializedText)

		if deserializedText == "" {
			t.Errorf("ISSUE FOUND: Content lost during JSON round-trip")

			// Debug the message structure
			msgJSON, _ := json.MarshalIndent(deserializedResponse.Choices[0].Message, "", "  ")
			t.Logf("Deserialized message structure:\n%s", string(msgJSON))
		} else {
			assert.Equal(t, "Hello World", deserializedText)
		}
	})

	t.Run("timing_simulation", func(t *testing.T) {
		// Test if timing affects content extraction
		tracer := NewTracingMiddleware("timing-test")

		ctx := context.Background()
		req := &ChatRequest{
			Messages: []Message{
				NewTextMessage(RoleUser, "Hello"),
			},
		}

		// Process immediately
		processedResp, err := tracer.ProcessResponse(ctx, req, originalResponse, nil)
		require.NoError(t, err)

		// Check both the returned response and what was captured
		assert.Equal(t, "Hello World", processedResp.Choices[0].Message.GetText())

		capturedContent := tracer.GetLastCapturedContent()
		choices := capturedContent["choices"].([]map[string]interface{})

		if choices[0]["text_preview"] == "" {
			t.Errorf("ISSUE FOUND: Content extraction returned empty even with direct processing")
		}
	})
}

// TestMessageContentStructure verifies the Message content structure
func TestMessageContentStructure(t *testing.T) {
	t.Run("text_message_creation", func(t *testing.T) {
		msg := NewTextMessage(RoleAssistant, "Hello World")

		// Verify basic properties
		assert.Equal(t, RoleAssistant, msg.Role)
		assert.Len(t, msg.Content, 1)
		assert.Equal(t, "Hello World", msg.GetText())

		// Check content structure
		content := msg.Content[0]
		assert.Equal(t, MessageTypeText, content.Type())

		// Verify JSON marshaling preserves content
		msgJSON, err := json.Marshal(msg)
		require.NoError(t, err)

		t.Logf("Message JSON: %s", string(msgJSON))

		// Unmarshal and verify content survives
		var unmarshaled Message
		err = json.Unmarshal(msgJSON, &unmarshaled)
		require.NoError(t, err)

		assert.Equal(t, "Hello World", unmarshaled.GetText())
	})

	t.Run("empty_content_scenarios", func(t *testing.T) {
		// Test various scenarios that might lead to empty content

		// Empty message
		emptyMsg := Message{Role: RoleAssistant}
		assert.Empty(t, emptyMsg.GetText())

		// Message with empty content array
		emptyContentMsg := Message{
			Role:    RoleAssistant,
			Content: []MessageContent{},
		}
		assert.Empty(t, emptyContentMsg.GetText())

		// Message with nil content
		nilContentMsg := Message{Role: RoleAssistant, Content: nil}
		assert.Empty(t, nilContentMsg.GetText())
	})
}
