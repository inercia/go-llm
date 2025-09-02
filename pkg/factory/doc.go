// Package factory provides provider registration and factory functionality for the go-llm framework.
//
// This package manages the registration of LLM providers and provides factory methods
// to create clients. When imported, it automatically registers all available providers
// through the side effects of importing their packages.
//
// Key components:
//   - Provider registration system with thread-safe registry
//   - Factory for creating clients based on configuration
//   - Automatic import of all available providers
//
// Example usage:
//
//	import (
//	    "github.com/inercia/go-llm/pkg/llm"
//	    "github.com/inercia/go-llm/pkg/factory"
//	)
//
//	factory := factory.New()
//	client, err := factory.CreateClient(llm.ClientConfig{
//	    Provider: "openai",
//	    Model: "gpt-4",
//	    APIKey: "your-api-key",
//	})
package factory
