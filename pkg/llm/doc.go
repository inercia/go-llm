// Package llm provides abstractions and interfaces for Large Language Model clients.
//
// This package defines the core interfaces that all LLM providers must implement,
// along with common types for requests, responses, messages, and streaming.
//
// The main components include:
//
// - Client interface: Core LLM client functionality
// - StreamingClient interface: Extended interface for streaming with tool injection
// - Message types: Multi-modal message support (text, images, files)
// - Tool system: Function calling and tool execution
// - Configuration: Provider-agnostic configuration
// - Error handling: Standardized error types
// - Streaming: Real-time response streaming with tool integration
//
// Provider implementations are located in separate packages under /pkg/providers/
// to maintain clean separation of concerns and avoid import cycles.
package llm
