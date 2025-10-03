package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/inercia/go-llm/pkg/factory"
	"github.com/inercia/go-llm/pkg/llm"
)

// Example response structures for different use cases
type AnalysisResult struct {
	Sentiment    string   `json:"sentiment" description:"The overall sentiment (positive, negative, neutral)"`
	Confidence   float64  `json:"confidence" minimum:"0" maximum:"1" description:"Confidence score between 0 and 1"`
	Keywords     []string `json:"keywords" description:"Important keywords extracted from the text"`
	Summary      string   `json:"summary" maxLength:"200" description:"Brief summary of the content"`
	Emoticons    []string `json:"emoticons,omitempty" description:"Detected emoticons or emojis"`
	Language     string   `json:"language" description:"Detected language code (e.g., 'en', 'es')"`
	WordCount    int      `json:"word_count" minimum:"0" description:"Number of words in the input"`
	HasQuestions bool     `json:"has_questions" description:"Whether the text contains questions"`
}

type MathProblemSolution struct {
	Steps []struct {
		Explanation string `json:"explanation" description:"Explanation of this step"`
		Operation   string `json:"operation" description:"The mathematical operation performed"`
		Result      string `json:"result" description:"Result after this step"`
	} `json:"steps" description:"Step-by-step solution breakdown"`
	FinalAnswer string `json:"final_answer" description:"The final numerical answer"`
	Method      string `json:"method" description:"The mathematical method used (e.g., 'substitution', 'elimination')"`
	Difficulty  string `json:"difficulty" enum:"easy,medium,hard" description:"Difficulty level of the problem"`
}

type ProductRecommendation struct {
	Products []struct {
		Name        string  `json:"name" description:"Product name"`
		Price       float64 `json:"price" minimum:"0" description:"Product price in USD"`
		Rating      float64 `json:"rating" minimum:"1" maximum:"5" description:"Average customer rating"`
		Category    string  `json:"category" description:"Product category"`
		InStock     bool    `json:"in_stock" description:"Whether the product is currently in stock"`
		Description string  `json:"description" maxLength:"500" description:"Product description"`
	} `json:"products" minItems:"1" maxItems:"5" description:"List of recommended products"`
	TotalBudget        float64  `json:"total_budget" minimum:"0" description:"Total budget for all recommendations"`
	Reasoning          string   `json:"reasoning" description:"Explanation for why these products were recommended"`
	AlternativeOptions []string `json:"alternative_options,omitempty" description:"Alternative product categories to consider"`
}

func main() {
	// Automatically detect and configure LLM provider from environment variables
	// This function checks for API keys in priority order and selects the best available provider
	config := llm.GetLLMFromEnv()

	// Override model if needed for structured outputs (some models work better)
	switch config.Provider {
	case "openai":
		config.Model = "gpt-4o-2024-08-06" // Use a model that supports structured outputs
	case "gemini":
		config.Model = "gemini-1.5-pro" // Gemini Pro for better structured output support
	}

	factory := factory.New()
	client, err := factory.CreateClient(config)
	if err != nil {
		log.Fatal("Failed to create client:", err)
	}
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	// Example 1: Text Analysis with JSON Schema
	fmt.Println("=== Example 1: Text Analysis with Structured Output ===")
	err = demonstrateTextAnalysis(ctx, client, config.Model)
	if err != nil {
		log.Printf("Text analysis example failed: %v", err)
	}

	// Example 2: Math Problem Solving
	fmt.Println("\n=== Example 2: Math Problem Solving ===")
	err = demonstrateMathSolving(ctx, client, config.Model)
	if err != nil {
		log.Printf("Math solving example failed: %v", err)
	}

	// Example 3: Product Recommendations
	fmt.Println("\n=== Example 3: Product Recommendations ===")
	err = demonstrateProductRecommendations(ctx, client, config.Model)
	if err != nil {
		log.Printf("Product recommendations example failed: %v", err)
	}

	// Example 4: Basic JSON Mode (without schema)
	fmt.Println("\n=== Example 4: Basic JSON Mode ===")
	err = demonstrateBasicJSONMode(ctx, client, config.Model)
	if err != nil {
		log.Printf("Basic JSON mode example failed: %v", err)
	}
}

func demonstrateTextAnalysis(ctx context.Context, client llm.Client, model string) error {
	// Generate schema from Go struct using the swaggest library
	responseFormat, err := llm.NewJSONSchemaResponseFormatStrictFromStruct(
		"text_analysis",
		"Analysis of text including sentiment, keywords, and other insights",
		AnalysisResult{},
	)
	if err != nil {
		return fmt.Errorf("failed to create response format: %w", err)
	}

	// Create request with structured output
	req := llm.ChatRequest{
		Model: model,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleSystem, "You are an expert text analyst. Analyze the given text and provide detailed insights."),
			llm.NewTextMessage(llm.RoleUser, "Please analyze this product review: 'I absolutely love this new smartphone! 📱 The camera quality is amazing and the battery lasts all day. However, I wish it was a bit cheaper. Overall, great purchase! 5/5 stars ⭐'"),
		},
		ResponseFormat: responseFormat,
	}

	// Make the request
	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("API call failed: %w", err)
	}

	// Extract and validate the JSON response
	var analysis AnalysisResult
	err = llm.ExtractAndValidateJSONToStruct(
		resp.Choices[0].Message.GetText(),
		&analysis,
		responseFormat.JSONSchema.Schema,
	)
	if err != nil {
		return fmt.Errorf("failed to extract structured response: %w", err)
	}

	// Display results
	fmt.Printf("Sentiment: %s (%.2f confidence)\n", analysis.Sentiment, analysis.Confidence)
	fmt.Printf("Keywords: %v\n", analysis.Keywords)
	fmt.Printf("Summary: %s\n", analysis.Summary)
	fmt.Printf("Language: %s\n", analysis.Language)
	fmt.Printf("Word Count: %d\n", analysis.WordCount)
	fmt.Printf("Has Questions: %t\n", analysis.HasQuestions)
	if len(analysis.Emoticons) > 0 {
		fmt.Printf("Emoticons: %v\n", analysis.Emoticons)
	}

	return nil
}

func demonstrateMathSolving(ctx context.Context, client llm.Client, model string) error {
	// Create schema for math problem solutions
	responseFormat, err := llm.NewJSONSchemaResponseFormatStrictFromStruct(
		"math_solution",
		"Step-by-step solution to a mathematical problem",
		MathProblemSolution{},
	)
	if err != nil {
		return fmt.Errorf("failed to create response format: %w", err)
	}

	req := llm.ChatRequest{
		Model: model,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleSystem, "You are a mathematics tutor. Provide step-by-step solutions to math problems."),
			llm.NewTextMessage(llm.RoleUser, "Solve this system of equations:\n2x + 3y = 7\n4x - y = 5"),
		},
		ResponseFormat: responseFormat,
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("API call failed: %w", err)
	}

	var solution MathProblemSolution
	err = llm.ExtractAndValidateJSONToStruct(
		resp.Choices[0].Message.GetText(),
		&solution,
		responseFormat.JSONSchema.Schema,
	)
	if err != nil {
		return fmt.Errorf("failed to extract structured response: %w", err)
	}

	// Display solution
	fmt.Printf("Method: %s\n", solution.Method)
	fmt.Printf("Difficulty: %s\n", solution.Difficulty)
	fmt.Printf("Final Answer: %s\n", solution.FinalAnswer)
	fmt.Println("Steps:")
	for i, step := range solution.Steps {
		fmt.Printf("  %d. %s\n", i+1, step.Explanation)
		fmt.Printf("     Operation: %s\n", step.Operation)
		fmt.Printf("     Result: %s\n", step.Result)
	}

	return nil
}

func demonstrateProductRecommendations(ctx context.Context, client llm.Client, model string) error {
	responseFormat, err := llm.NewJSONSchemaResponseFormatFromStruct(
		"product_recommendations",
		"Product recommendations based on user preferences",
		ProductRecommendation{},
	)
	if err != nil {
		return fmt.Errorf("failed to create response format: %w", err)
	}

	req := llm.ChatRequest{
		Model: model,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleSystem, "You are a product recommendation expert. Suggest products based on user needs and budget."),
			llm.NewTextMessage(llm.RoleUser, "I need recommendations for home office equipment. My budget is $500 and I work from home doing software development. I need good ergonomics and productivity tools."),
		},
		ResponseFormat: responseFormat,
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("API call failed: %w", err)
	}

	var recommendations ProductRecommendation
	err = llm.ExtractJSONToStruct(resp.Choices[0].Message.GetText(), &recommendations)
	if err != nil {
		return fmt.Errorf("failed to extract structured response: %w", err)
	}

	// Display recommendations
	fmt.Printf("Total Budget: $%.2f\n", recommendations.TotalBudget)
	fmt.Printf("Reasoning: %s\n", recommendations.Reasoning)
	fmt.Println("Recommended Products:")
	for i, product := range recommendations.Products {
		fmt.Printf("  %d. %s - $%.2f\n", i+1, product.Name, product.Price)
		fmt.Printf("     Category: %s | Rating: %.1f/5 | In Stock: %t\n",
			product.Category, product.Rating, product.InStock)
		fmt.Printf("     %s\n", product.Description)
	}

	if len(recommendations.AlternativeOptions) > 0 {
		fmt.Printf("Alternative Options: %v\n", recommendations.AlternativeOptions)
	}

	return nil
}

func demonstrateBasicJSONMode(ctx context.Context, client llm.Client, model string) error {
	// Use basic JSON mode without a specific schema
	req := llm.ChatRequest{
		Model: model,
		Messages: []llm.Message{
			llm.NewTextMessage(llm.RoleUser, "Create a simple JSON object with information about the planet Mars, including name, type, distance from sun, and three interesting facts."),
		},
		ResponseFormat: llm.NewJSONResponseFormat(),
	}

	resp, err := client.ChatCompletion(ctx, req)
	if err != nil {
		return fmt.Errorf("API call failed: %w", err)
	}

	// Extract JSON and pretty print it
	jsonStr := llm.ExtractJSONFromResponse(resp.Choices[0].Message.GetText())

	// Parse and reformat for pretty printing
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	prettyJSON, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to format JSON: %w", err)
	}

	fmt.Println("Generated JSON:")
	fmt.Println(string(prettyJSON))

	return nil
}
