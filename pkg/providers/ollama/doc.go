// Package ollama provides an Ollama client implementation for the go-llm library.
//
// This package implements the llm.Client interface for Ollama's local LLM hosting,
// supporting chat completions, streaming, and various open-source models.
//
// Features:
// - Local model hosting via Ollama
// - Streaming chat completions
// - Multiple model support (Llama, Mistral, CodeLlama, etc.)
// - Automatic model detection and configuration
// - Multi-modal content (text, images)
//
// The client connects to a local Ollama instance running on localhost:11434
// by default, but can be configured to use any Ollama endpoint.
package ollama
