package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/inercia/go-llm/pkg/factory"
	"github.com/inercia/go-llm/pkg/llm"
)

func main() {
	// Create factory
	factory := factory.New()

	// Create OpenAI client (replace with your API key)
	client, err := factory.CreateClient(llm.ClientConfig{
		Provider: "openai",
		Model:    "gpt-3.5-turbo",
		APIKey:   "your-openai-api-key", // Replace with actual key
	})
	if err != nil {
		log.Fatal("Failed to create client:", err)
	}
	defer func() { _ = client.Close() }()

	// Check streaming support
	modelInfo := client.GetModelInfo()
	if !modelInfo.SupportsStreaming {
		log.Fatal("Model does not support streaming")
	}

	// Create streaming request
	req := llm.ChatRequest{
		Model: modelInfo.Name,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("Tell me a short story about a robot learning to dance.")}},
		},
		Stream: true,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stream, err := client.StreamChatCompletion(ctx, req)
	if err != nil {
		log.Fatal("Failed to create stream:", err)
	}

	var fullStory strings.Builder
	fmt.Print("Robot's story: ")
	for event := range stream {
		switch {
		case event.IsDelta():
			if len(event.Choice.Delta.Content) > 0 {
				if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
					chunk := textContent.GetText()
					fullStory.WriteString(chunk)
					fmt.Print(chunk)
				}
			}
		case event.IsDone():
			fmt.Println("\n\nStory complete! Finish reason:", event.Choice.FinishReason)
			fmt.Println("Full story:", fullStory.String())
		case event.IsError():
			log.Printf("Stream error: %s (code: %s)", event.Error.Message, event.Error.Code)
		}
	}

	// Non-streaming example for comparison
	fmt.Println("\n\n--- Non-Streaming Example ---")
	nonStreamReq := llm.ChatRequest{
		Model: modelInfo.Name,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.MessageContent{llm.NewTextContent("What is the moral of the story?")}},
		},
	}
	resp, err := client.ChatCompletion(ctx, nonStreamReq)
	if err != nil {
		log.Fatal("Non-streaming request failed:", err)
	}

	if len(resp.Choices) > 0 {
		fmt.Println("Moral:", resp.Choices[0].Message.GetText())
		fmt.Printf("Tokens used: %d prompt + %d completion = %d total\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	}
}
