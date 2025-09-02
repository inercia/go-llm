package llm

import "regexp"

// ModelRegistry provides centralized model capability information
type ModelRegistry struct{}

// ModelCapabilities defines what a model can do
type ModelCapabilities struct {
	MaxTokens         int  `json:"max_tokens"`
	SupportsTools     bool `json:"supports_tools"`
	SupportsVision    bool `json:"supports_vision"`
	SupportsFiles     bool `json:"supports_files"`
	SupportsStreaming bool `json:"supports_streaming"`
}

type ModelEntry struct {
	Pattern      *regexp.Regexp
	Capabilities ModelCapabilities
}

var modelData = map[string][]ModelEntry{
	"openrouter": []ModelEntry{
		{
			Pattern: regexp.MustCompile(`^(openai/gpt-4o|openai/gpt-4o-mini)`),
			Capabilities: ModelCapabilities{
				MaxTokens:         128000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^openai/gpt-4-turbo`),
			Capabilities: ModelCapabilities{
				MaxTokens:         128000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^openai/gpt-4`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^openai/gpt-3\.5-turbo`),
			Capabilities: ModelCapabilities{
				MaxTokens:         16384,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^anthropic/claude-3`),
			Capabilities: ModelCapabilities{
				MaxTokens:         200000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^anthropic/claude-2`),
			Capabilities: ModelCapabilities{
				MaxTokens:         100000,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^google/gemini-1\.5-pro`),
			Capabilities: ModelCapabilities{
				MaxTokens:         2000000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^google/gemini-1\.5-flash`),
			Capabilities: ModelCapabilities{
				MaxTokens:         1000000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^meta-llama/llama-3`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^mistralai/mistral-`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         4096,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
	},
	"openai": []ModelEntry{
		{
			Pattern: regexp.MustCompile(`^(gpt-4|gpt-4-0613)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^(gpt-4-32k|gpt-4-32k-0613)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^(gpt-4o|gpt-4o-mini)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         16384,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^gpt-4-vision-preview$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     false,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^(gpt-4-turbo|gpt-4-turbo-preview)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         128000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^(gpt-3\.5-turbo|gpt-3\.5-turbo-16k)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         16384,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         4096,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
	},
	"gemini": []ModelEntry{
		{
			Pattern: regexp.MustCompile(`^gemini-1\.5-pro$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         2000000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^(gemini-1\.5-flash|gemini-1\.5-flash-8b)$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         1000000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^gemini-2\.0-flash-exp$`),
			Capabilities: ModelCapabilities{
				MaxTokens:         1000000,
				SupportsTools:     true,
				SupportsVision:    true,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         30720,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     true,
				SupportsStreaming: true,
			},
		},
	},
	"ollama": []ModelEntry{
		{
			Pattern: regexp.MustCompile(`.*llama3\.1.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         131072,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*llama3.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*llama2.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         4096,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*codellama.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         16384,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*mistral.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         8192,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*qwen.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         4096,
				SupportsTools:     false,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
	},
	"deepseek": []ModelEntry{
		{
			Pattern: regexp.MustCompile(`^deepseek-chat`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^deepseek-coder`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     true,
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
		{
			Pattern: regexp.MustCompile(`^deepseek-.*`),
			Capabilities: ModelCapabilities{
				MaxTokens:         32768,
				SupportsTools:     false, // Conservative default
				SupportsVision:    false,
				SupportsFiles:     false,
				SupportsStreaming: true,
			},
		},
	},
}

// GetModelCapabilities returns capabilities for a given model and provider
func (r *ModelRegistry) GetModelCapabilities(provider, model string) ModelCapabilities {
	if entries, ok := modelData[provider]; ok {
		for _, entry := range entries {
			if entry.Pattern.MatchString(model) {
				return entry.Capabilities
			}
		}
		// Fallback for provider but no matching model (though .* ensures match)
		switch provider {
		case "openrouter":
			return ModelCapabilities{MaxTokens: 4096, SupportsTools: false, SupportsVision: false, SupportsFiles: false, SupportsStreaming: true}
		case "openai":
			return ModelCapabilities{MaxTokens: 4096, SupportsTools: false, SupportsVision: false, SupportsFiles: true, SupportsStreaming: true}
		case "gemini":
			return ModelCapabilities{MaxTokens: 30720, SupportsTools: true, SupportsVision: false, SupportsFiles: true, SupportsStreaming: true}
		case "ollama":
			return ModelCapabilities{MaxTokens: 4096, SupportsTools: false, SupportsVision: false, SupportsFiles: false, SupportsStreaming: true}
		case "deepseek":
			return ModelCapabilities{MaxTokens: 32768, SupportsTools: false, SupportsVision: false, SupportsFiles: false, SupportsStreaming: true}
		default:
			return ModelCapabilities{MaxTokens: 4096, SupportsTools: false, SupportsVision: false}
		}
	}
	return ModelCapabilities{
		MaxTokens:      4096,
		SupportsTools:  false,
		SupportsVision: false,
	}
}

// Global model registry instance
var modelRegistry = &ModelRegistry{}
