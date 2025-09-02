// Model information and capabilities
package llm

// ModelInfo contains information about the model
type ModelInfo struct {
	Name              string `json:"name"`
	Provider          string `json:"provider"`
	MaxTokens         int    `json:"max_tokens"`
	SupportsTools     bool   `json:"supports_tools"`
	SupportsVision    bool   `json:"supports_vision"`
	SupportsFiles     bool   `json:"supports_files"`
	SupportsStreaming bool   `json:"supports_streaming"`
}
