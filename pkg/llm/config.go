// Configuration types and response format specifications
package llm

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

const (
	DefaultOpenAIModel     = "gpt-4o-mini"
	DefaultGeminiModel     = "gemini-1.5-flash"
	DefaultDeepSeekModel   = "deepseek-chat"
	DefaultOpenRouterModel = "meta-llama/llama-3.1-8b-instruct:free"
	DefaultBedrockModel    = "anthropic.claude-3-haiku-20240307-v1:0"
	DefaultOllamaModel     = "gpt-oss:20b"
)

const DefaultOllamaBaseURL = "http://localhost:11434"

const (
	DefaultOllamaTimeout      = 60 * time.Second
	DefaultOllamaQuickTimeout = 30 * time.Second
)

// ClientConfig holds configuration for creating LLM clients
type ClientConfig struct {
	Provider   string            `json:"provider"` // openai, gemini, ollama, anthropic, etc.
	Model      string            `json:"model"`
	APIKey     string            `json:"api_key,omitempty"`
	BaseURL    string            `json:"base_url,omitempty"`
	Timeout    time.Duration     `json:"timeout,omitempty"`
	MaxRetries int               `json:"max_retries,omitempty"`
	Extra      map[string]string `json:"extra,omitempty"` // Provider-specific configs
}

// ResponseFormat specifies the desired response format for structured outputs
type ResponseFormat struct {
	Type       ResponseFormatType `json:"type"`
	JSONSchema *JSONSchema        `json:"json_schema,omitempty"`
}

// ResponseFormatType defines the type of response format
type ResponseFormatType string

const (
	// ResponseFormatText indicates plain text response (default)
	ResponseFormatText ResponseFormatType = "text"
	// ResponseFormatJSON indicates JSON object response without strict schema
	ResponseFormatJSON ResponseFormatType = "json_object"
	// ResponseFormatJSONSchema indicates JSON response with strict schema validation
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// JSONSchema represents a JSON Schema specification for structured outputs
type JSONSchema struct {
	Name        string      `json:"name,omitempty"`        // Schema name (required by some providers)
	Description string      `json:"description,omitempty"` // Human-readable description
	Schema      interface{} `json:"schema"`                // The actual JSON Schema object
	Strict      *bool       `json:"strict,omitempty"`      // Enable strict validation (OpenAI-specific)
}

// parseTimeoutFromEnv parses timeout from environment variable with fallback to default
func parseTimeoutFromEnv(envVar string, defaultTimeout time.Duration) time.Duration {
	if timeoutStr := os.Getenv(envVar); timeoutStr != "" {
		if timeoutSecs, err := strconv.Atoi(timeoutStr); err == nil && timeoutSecs > 0 {
			return time.Duration(timeoutSecs) * time.Second
		}
	}
	return defaultTimeout
}

func GetLLMFromEnv() ClientConfig {
	// Priority 1: Custom OpenAI-compatible endpoint (highest priority if explicitly configured)
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		fmt.Println("🔑 Using Custom OpenAI-compatible API")
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			apiKey = "dummy" // Some endpoints don't require real keys
		}

		model := DefaultOpenAIModel
		// Allow model override for custom endpoints
		if customModel := os.Getenv("OPENAI_MODEL"); customModel != "" {
			model = customModel
		} else if customModel := os.Getenv("MODEL"); customModel != "" {
			model = customModel
		}

		return ClientConfig{
			Provider: "openai",
			Model:    model,
			APIKey:   apiKey,
			BaseURL:  baseURL,
			Timeout:  parseTimeoutFromEnv("OPENAI_TIMEOUT", 30*time.Second),
		}
	}

	// Priority 2: OpenAI API
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		fmt.Println("🔑 Using OpenAI API")
		return ClientConfig{
			Provider: "openai",
			Model:    DefaultOpenAIModel, // Cost-effective, high-quality model
			APIKey:   apiKey,
			Timeout:  parseTimeoutFromEnv("OPENAI_TIMEOUT", 30*time.Second),
		}
	}

	// Priority 3: Gemini API
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		fmt.Println("🔑 Using Gemini API")
		model := DefaultGeminiModel // Fast and cost-effective

		// Allow model override via environment variable
		if customModel := os.Getenv("GEMINI_MODEL"); customModel != "" {
			model = customModel
		}

		return ClientConfig{
			Provider: "gemini",
			Model:    model,
			APIKey:   apiKey,
			Timeout:  parseTimeoutFromEnv("GEMINI_TIMEOUT", 30*time.Second),
		}
	}

	// Priority 4: DeepSeek API
	if apiKey := os.Getenv("DEEPSEEK_API_KEY"); apiKey != "" {
		fmt.Println("🔑 Using DeepSeek API")
		model := DefaultDeepSeekModel

		// Allow model override via environment variable
		if customModel := os.Getenv("DEEPSEEK_MODEL"); customModel != "" {
			model = customModel
		}

		return ClientConfig{
			Provider: "deepseek",
			Model:    model,
			APIKey:   apiKey,
			Timeout:  parseTimeoutFromEnv("DEEPSEEK_TIMEOUT", 30*time.Second),
		}
	}

	// Priority 5: OpenRouter API
	if apiKey := os.Getenv("OPENROUTER_API_KEY"); apiKey != "" {
		fmt.Println("🔑 Using OpenRouter API")
		model := DefaultOpenRouterModel

		// Allow model override via environment variable
		if customModel := os.Getenv("OPENROUTER_MODEL"); customModel != "" {
			model = customModel
		}

		return ClientConfig{
			Provider: "openrouter",
			Model:    model,
			APIKey:   apiKey,
			Timeout:  parseTimeoutFromEnv("OPENROUTER_TIMEOUT", 30*time.Second),
		}
	}

	// Priority 6: AWS Bedrock (uses AWS credential chain)
	if os.Getenv("AWS_ACCESS_KEY_ID") != "" || os.Getenv("AWS_PROFILE") != "" || os.Getenv("AWS_BEDROCK_MODEL") != "" {
		fmt.Println("🔑 Using AWS Bedrock")
		model := DefaultBedrockModel

		// Allow model override via environment variable
		if customModel := os.Getenv("AWS_BEDROCK_MODEL"); customModel != "" {
			model = customModel
		} else if customModel := os.Getenv("BEDROCK_MODEL"); customModel != "" {
			model = customModel
		}

		// Get AWS region from environment or use default
		region := os.Getenv("AWS_REGION")
		if region == "" {
			region = "us-east-1"
		}

		config := ClientConfig{
			Provider: "bedrock",
			Model:    model,
			Timeout:  parseTimeoutFromEnv("AWS_BEDROCK_TIMEOUT", 60*time.Second),
		}

		// Set region and endpoints in Extra field
		if config.Extra == nil {
			config.Extra = make(map[string]string)
		}
		config.Extra["region"] = region

		// Add custom endpoints if specified
		if bedrockEndpoint := os.Getenv("AWS_BEDROCK_ENDPOINT"); bedrockEndpoint != "" {
			config.Extra["bedrock_endpoint"] = bedrockEndpoint
		}
		if bedrockRuntimeEndpoint := os.Getenv("AWS_BEDROCK_RUNTIME_ENDPOINT"); bedrockRuntimeEndpoint != "" {
			config.Extra["bedrock_runtime_endpoint"] = bedrockRuntimeEndpoint
		}
		// Support generic BEDROCK_ENDPOINT for runtime endpoint (most common use case)
		if bedrockEndpoint := os.Getenv("BEDROCK_ENDPOINT"); bedrockEndpoint != "" {
			config.Extra["base_url"] = bedrockEndpoint
		}

		return config
	}

	model := DefaultOllamaModel
	baseURL := DefaultOllamaBaseURL

	// Default: Ollama (local, free)
	fmt.Printf("🔑 Using Ollama (local) at %s\n", baseURL)
	fmt.Println("💡 To use cloud providers: set OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, or configure AWS credentials")

	return ClientConfig{
		Provider: "ollama",
		Model:    model,
		BaseURL:  baseURL,
		Timeout:  parseTimeoutFromEnv("OLLAMA_TIMEOUT", DefaultOllamaTimeout),
	}
}
