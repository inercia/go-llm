package test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inercia/go-llm/pkg/llm"
)

func TestToolsBasicFunctionality(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireToolSupport(t, client)

	ctx := context.Background()

	t.Run("simple_tool_call", func(t *testing.T) {
		// Define a simple calculator tool
		calculatorTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "calculate",
				Description: "Perform basic arithmetic operations",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"operation": map[string]interface{}{
							"type":        "string",
							"description": "The operation to perform",
							"enum":        []string{"add", "subtract", "multiply", "divide"},
						},
						"a": map[string]interface{}{
							"type":        "number",
							"description": "First number",
						},
						"b": map[string]interface{}{
							"type":        "number",
							"description": "Second number",
						},
					},
					"required": []string{"operation", "a", "b"},
				},
			},
		}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("I need to calculate 15 + 27. Please use the calculate tool."),
				}},
			},
			Tools: []llm.Tool{calculatorTool},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Tool-enabled chat should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		choice := resp.Choices[0]
		t.Logf("Response finish reason: %s", choice.FinishReason)

		// Check if the model wants to call a tool
		if len(choice.Message.ToolCalls) > 0 {
			toolCall := choice.Message.ToolCalls[0]
			t.Logf("Tool call detected: %s", toolCall.Function.Name)
			t.Logf("Tool arguments: %s", toolCall.Function.Arguments)

			assert.Equal(t, "calculate", toolCall.Function.Name, "Should call the calculate tool")

			// Parse the arguments to check they're reasonable
			var args map[string]interface{}
			err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			require.NoError(t, err, "Tool arguments should be valid JSON")

			assert.Contains(t, args, "operation", "Arguments should contain operation")
			assert.Contains(t, args, "a", "Arguments should contain first number")
			assert.Contains(t, args, "b", "Arguments should contain second number")

			// Check the operation makes sense
			operation, ok := args["operation"].(string)
			require.True(t, ok, "Operation should be a string")
			assert.Equal(t, "add", operation, "Should use add operation for 15 + 27")

			t.Logf("✅ Tool call successful: %s with args %v", toolCall.Function.Name, args)
		} else {
			// Some models might just respond with text instead of calling tools
			responseText := choice.Message.GetText()
			t.Logf("No tool call, got text response: %s", responseText)

			// If no tool call, the model should at least mention calculation or the result
			lowerResponse := strings.ToLower(responseText)
			hasCalculation := strings.Contains(lowerResponse, "42") ||
				strings.Contains(lowerResponse, "calculate") ||
				strings.Contains(lowerResponse, "add") ||
				strings.Contains(lowerResponse, "15") ||
				strings.Contains(lowerResponse, "27")

			assert.True(t, hasCalculation, "Response should mention calculation or numbers")
		}
	})

	t.Run("weather_tool_call", func(t *testing.T) {
		// Define a weather tool
		weatherTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "get_weather",
				Description: "Get current weather information for a location",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "The city and state/country, e.g. 'San Francisco, CA'",
						},
						"unit": map[string]interface{}{
							"type":        "string",
							"description": "Temperature unit",
							"enum":        []string{"celsius", "fahrenheit"},
						},
					},
					"required": []string{"location"},
				},
			},
		}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("What's the weather like in Paris, France? Please use the weather tool."),
				}},
			},
			Tools: []llm.Tool{weatherTool},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Weather tool chat should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		choice := resp.Choices[0]

		if len(choice.Message.ToolCalls) > 0 {
			toolCall := choice.Message.ToolCalls[0]
			t.Logf("Weather tool call: %s", toolCall.Function.Name)
			t.Logf("Arguments: %s", toolCall.Function.Arguments)

			assert.Equal(t, "get_weather", toolCall.Function.Name)

			var args map[string]interface{}
			err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
			require.NoError(t, err)

			location, exists := args["location"]
			assert.True(t, exists, "Location should be provided")

			locationStr, ok := location.(string)
			assert.True(t, ok, "Location should be a string")
			assert.Contains(t, strings.ToLower(locationStr), "paris", "Location should contain Paris")

			t.Logf("✅ Weather tool call successful for location: %s", locationStr)
		} else {
			responseText := choice.Message.GetText()
			t.Logf("No weather tool call, got response: %s", responseText)
		}
	})
}

func TestToolsMultipleTools(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireToolSupport(t, client)

	ctx := context.Background()

	t.Run("multiple_available_tools", func(t *testing.T) {
		// Define multiple tools
		calculatorTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "calculate",
				Description: "Perform arithmetic operations",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expression": map[string]interface{}{
							"type":        "string",
							"description": "Mathematical expression to evaluate",
						},
					},
					"required": []string{"expression"},
				},
			},
		}

		searchTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "web_search",
				Description: "Search the web for information",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "Search query",
						},
					},
					"required": []string{"query"},
				},
			},
		}

		timeTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "get_current_time",
				Description: "Get the current date and time",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("What time is it right now? Please use the appropriate tool."),
				}},
			},
			Tools: []llm.Tool{calculatorTool, searchTool, timeTool},
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Multiple tools chat should succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		choice := resp.Choices[0]

		if len(choice.Message.ToolCalls) > 0 {
			toolCall := choice.Message.ToolCalls[0]
			t.Logf("Selected tool: %s", toolCall.Function.Name)

			// Should choose the time tool for this question
			assert.Equal(t, "get_current_time", toolCall.Function.Name,
				"Should select the time tool for time-related question")

			t.Logf("✅ Correctly selected time tool from multiple options")
		} else {
			responseText := choice.Message.GetText()
			t.Logf("No tool call, got response: %s", responseText)

			// Should at least mention time
			lowerResponse := strings.ToLower(responseText)
			hasTimeInfo := strings.Contains(lowerResponse, "time") ||
				strings.Contains(lowerResponse, "clock") ||
				strings.Contains(lowerResponse, "current")
			assert.True(t, hasTimeInfo, "Response should be time-related")
		}
	})
}

func TestToolsConversationFlow(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireToolSupport(t, client)

	ctx := context.Background()

	t.Run("tool_call_with_result", func(t *testing.T) {
		// Define a simple tool
		factTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "get_fact",
				Description: "Get an interesting fact about a topic",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"topic": map[string]interface{}{
							"type":        "string",
							"description": "The topic to get a fact about",
						},
					},
					"required": []string{"topic"},
				},
			},
		}

		// First request asking for a fact
		req1 := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Tell me an interesting fact about dolphins. Use the get_fact tool."),
				}},
			},
			Tools: []llm.Tool{factTool},
		}

		resp1, err := client.ChatCompletion(ctx, req1)
		require.NoError(t, err, "First request should succeed")
		require.NotNil(t, resp1)
		require.Len(t, resp1.Choices, 1)

		choice1 := resp1.Choices[0]

		if len(choice1.Message.ToolCalls) > 0 {
			toolCall := choice1.Message.ToolCalls[0]
			t.Logf("Tool call made: %s(%s)", toolCall.Function.Name, toolCall.Function.Arguments)

			// Simulate tool execution result
			toolResult := "Dolphins can recognize themselves in mirrors, showing self-awareness similar to humans and great apes."

			// Continue conversation with tool result
			req2 := llm.ChatRequest{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.MessageContent{
						llm.NewTextContent("Tell me an interesting fact about dolphins. Use the get_fact tool."),
					}},
					choice1.Message, // Include the assistant's response with tool call
					{Role: llm.RoleTool, Content: []llm.MessageContent{
						llm.NewTextContent(toolResult),
					}, ToolCallID: toolCall.ID},
				},
				Tools: []llm.Tool{factTool},
			}

			resp2, err := client.ChatCompletion(ctx, req2)
			require.NoError(t, err, "Follow-up request should succeed")
			require.NotNil(t, resp2)
			require.Len(t, resp2.Choices, 1)

			finalResponse := resp2.Choices[0].Message.GetText()
			t.Logf("Final response after tool use: %s", finalResponse)

			// Should incorporate the tool result
			lowerResponse := strings.ToLower(finalResponse)
			hasDolphinInfo := strings.Contains(lowerResponse, "dolphin") ||
				strings.Contains(lowerResponse, "mirror") ||
				strings.Contains(lowerResponse, "self-aware")

			assert.True(t, hasDolphinInfo, "Response should incorporate dolphin fact from tool")

			t.Logf("✅ Tool conversation flow completed successfully")
		} else {
			t.Log("Model didn't call tool, skipping follow-up test")
		}
	})
}

func TestToolsErrorHandling(t *testing.T) {
	client := createTestClient(t)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireToolSupport(t, client)

	ctx := context.Background()

	t.Run("invalid_tool_definition", func(t *testing.T) {
		// Tool with missing required fields
		invalidTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name: "", // Missing name
				// Missing description and parameters
			},
		}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Use any available tool."),
				}},
			},
			Tools: []llm.Tool{invalidTool},
		}

		// This should either fail gracefully or ignore the invalid tool
		resp, err := client.ChatCompletion(ctx, req)
		if err != nil {
			t.Logf("Invalid tool properly caused error: %v", err)
			// Should be a proper LLM error
			if llmErr, ok := err.(*llm.Error); ok {
				assert.NotEmpty(t, llmErr.Message, "Error should have a message")
			}
		} else {
			t.Logf("Invalid tool handled gracefully: %s", resp.Choices[0].Message.GetText())
		}
	})

	t.Run("no_tools_available", func(t *testing.T) {
		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Please use a tool to help me."),
				}},
			},
			// No tools provided
		}

		resp, err := client.ChatCompletion(ctx, req)
		require.NoError(t, err, "Request without tools should still succeed")
		require.NotNil(t, resp)
		require.Len(t, resp.Choices, 1)

		responseText := resp.Choices[0].Message.GetText()
		require.NotEmpty(t, responseText, "Should get some response")

		// Should not have any tool calls
		assert.Nil(t, resp.Choices[0].Message.ToolCalls, "Should not have tool calls when no tools provided")

		t.Logf("No tools available response: %s", responseText)
	})
}

func TestToolsStreaming(t *testing.T) {
	client := createTestClientWithTimeout(t, 15*time.Second)
	defer func() { _ = client.Close() }()

	skipIfNoProvider(t, client)
	requireToolSupport(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	t.Run("streaming_with_tools", func(t *testing.T) {
		simpleTool := llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "get_info",
				Description: "Get information about a topic",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"topic": map[string]interface{}{
							"type":        "string",
							"description": "Topic to get info about",
						},
					},
					"required": []string{"topic"},
				},
			},
		}

		req := llm.ChatRequest{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.MessageContent{
					llm.NewTextContent("Get information about artificial intelligence using the available tool."),
				}},
			},
			Tools:  []llm.Tool{simpleTool},
			Stream: true,
		}

		stream, err := client.StreamChatCompletion(ctx, req)
		require.NoError(t, err, "Tool streaming should succeed")
		require.NotNil(t, stream, "Stream should not be nil")

		eventCount := 0
		hasToolCall := false
		var fullResponse strings.Builder

		for event := range stream {
			eventCount++
			t.Logf("Stream event %d: Type=%s", eventCount, event.Type)

			if event.IsError() {
				t.Errorf("Stream error: %v", event.Error)
				break
			} else if event.IsDelta() && event.Choice != nil && event.Choice.Delta != nil {
				// Check for tool calls in delta
				if len(event.Choice.Delta.ToolCalls) > 0 {
					hasToolCall = true
					toolCall := event.Choice.Delta.ToolCalls[0]
					if toolCall.Function != nil && toolCall.Function.Name != "" {
						t.Logf("Tool call in stream: %s", toolCall.Function.Name)
					} else {
						t.Logf("Tool call delta received (partial or empty)")
					}
				}

				// Extract text content
				for _, content := range event.Choice.Delta.Content {
					if textContent, ok := content.(*llm.TextContent); ok {
						fullResponse.WriteString(textContent.GetText())
					}
				}
			} else if event.IsDone() {
				t.Logf("Stream completed: %s", event.Choice.FinishReason)
				break
			}

			// Safety limit
			if eventCount >= 50 {
				t.Log("Reached event limit")
				break
			}
		}

		require.Greater(t, eventCount, 0, "Should receive streaming events")

		response := fullResponse.String()
		t.Logf("Streaming response (%d events, tool call: %t): %s",
			eventCount, hasToolCall, response)

		// Either should have a tool call or some text response
		hasContent := hasToolCall || len(response) > 0
		assert.True(t, hasContent, "Should have either tool call or text response")
	})
}
