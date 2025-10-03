# Documentation Index

Welcome to the documentation for the Flexible, multi-provider LLM client library for Go.

This folder contains guides and references for using the library effectively.

## Quick Start

For basic setup and code examples, see [Usage Guide](usage.md).

For more advanced topics, see [this document](advanced.md).

## Core Features

Learn how to use the library's key capabilities:

- **[Streaming](streaming.md)**: Real-time response generation with streaming chat completions
- **[Tools & Function Calling](tools.md)**: Enable LLMs to call external functions and APIs
- **[Multimodal Messages](multimodal.md)**: Work with images, files, and mixed content types

## Architecture Overview

Understand the library's design, components, and how they interact in [Architecture Overview](architecture.md).

## Providers

Detailed information on each supported LLM provider, including features, setup, limitations, and troubleshooting:

- [AWS Bedrock](providers/bedrock.md): Claude, Titan, and Llama models via AWS Bedrock (cloud-based, AWS account required).
- [Gemini](providers/gemini.md): Google Gemini integration (cloud-based).
- [OpenAI](providers/openai.md): GPT models via official API (cloud-based, paid).
- [OpenRouter](providers/openrouter.md): Multi-provider API access (cloud-based, pay-per-use).
- [Ollama](providers/ollama.md): Local models via Ollama server (offline).

## Additional Resources

- **Examples**: Check the [examples/](../examples/) directory for runnable code demonstrating streaming, tools, and multimodal usage.
- **API Reference**: The root [README.md](../README.md) covers architecture, testing, and advanced usage.
- **Package Docs**: Run `go doc ./pkg/llm` for GoDoc-style reference.

If you're contributing or need more details, refer to the [AGENTS.md](../AGENTS.md) for development guidelines.

For issues or questions, open a GitHub issue.
