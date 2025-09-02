// Error types and handling
package llm

// Error represents a standardized LLM error
type Error struct {
	Code       string `json:"code"`
	Message    string `json:"message"`
	Type       string `json:"type"`
	StatusCode int    `json:"status_code,omitempty"`
}

func (e *Error) Error() string {
	return e.Message
}
