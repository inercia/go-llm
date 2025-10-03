// Package bedrock provides AWS Bedrock integration for the go-llm framework.
//
// This package implements the llm.Client interface for AWS Bedrock, enabling
// unified access to multiple foundation models including Claude (Anthropic),
// Titan (Amazon), and Llama (Meta) through a single consistent API.
//
// The client automatically handles model-specific request/response formats
// and provides features like streaming chat completions, multi-modal support
// (for compatible models), and proper error mapping.
//
// Key features:
//   - Support for multiple model families (Claude, Titan, Llama)
//   - Automatic format conversion based on model type
//   - Streaming and non-streaming chat completions
//   - Multi-modal support for Claude 3 models
//   - Health checks and error standardization
//   - Regional configuration support
//
// Usage:
//
//	client, err := bedrock.NewClient(llm.ClientConfig{
//	    Provider: "bedrock",
//	    Model:    "anthropic.claude-3-sonnet-20240229-v1:0",
//	    Extra: map[string]string{
//	        "region": "us-east-1",
//	    },
//	})
//
// The client uses the AWS SDK's default credential chain for authentication,
// supporting environment variables, IAM roles, profiles, and other standard
// AWS authentication methods.
package bedrock
