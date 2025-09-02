// Package openai provides an OpenAI client implementation for the go-llm library.
//
// This package implements the llm.Client interface for OpenAI's GPT models,
// supporting chat completions, streaming, tools (function calling), and
// multi-modal inputs including text, images, and files.
//
// Features:
// - Full GPT model support (GPT-3.5, GPT-4, GPT-4 Vision, etc.)
// - Streaming chat completions
// - Function calling and tool execution
// - Multi-modal content (text, images, files)
// - JSON mode and structured output
// - Automatic model selection for multi-modal content
//
// The client automatically handles provider-specific request/response
// transformations while maintaining compatibility with the common llm interfaces.
package openai
