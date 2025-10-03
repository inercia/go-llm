# Multimodal Messages

This guide covers how to work with multimodal messages in the Go LLM library. Multimodal support enables you to send images, files, and mixed content types alongside text in your LLM interactions.

## Overview

The library supports multiple content types within a single message:

- **Text Content**: Standard text messages
- **Image Content**: Images in various formats (JPEG, PNG, GIF, WebP, etc.)
- **File Content**: Text-based files (JSON, CSV, TXT, XML, etc.)
- **Mixed Content**: Combining multiple content types in one message

This enables rich interactions where you can analyze images, process documents, and provide comprehensive context to language models.

## Content Types

### Text Content

The most basic content type for regular text messages:

```go
import "github.com/inercia/go-llm/pkg/llm"

// Create text content
textContent := llm.NewTextContent("Describe what you see in this image.")

// Use in a message
message := llm.Message{
    Role:    llm.RoleUser,
    Content: []llm.MessageContent{textContent},
}
```

### Image Content

For sending images to vision-capable models:

```go
// From file bytes
imageData, err := os.ReadFile("path/to/image.jpg")
if err != nil {
    log.Fatal("Failed to read image:", err)
}

imageContent := llm.NewImageContentFromBytes(imageData, "image/jpeg")

// From base64 string (note: this helper doesn't exist, use URL format instead)
base64Image := "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
imageContent := llm.NewImageContentFromURL(base64Image, "image/jpeg")

// From URL (if supported by provider)
imageContent := llm.NewImageContentFromURL("https://example.com/image.jpg", "image/jpeg")
```

### File Content

For processing text-based documents:

```go
// Read file data
fileData, err := os.ReadFile("document.json")
if err != nil {
    log.Fatal("Failed to read file:", err)
}

fileContent := llm.NewFileContentFromBytes(
    fileData,
    "document.json",
    "application/json",
)

// Or from string
csvData := "name,age,city\nAlice,30,New York\nBob,25,London"
fileContent := llm.NewFileContentFromBytes(
    []byte(csvData),
    "data.csv",
    "text/csv",
)
```

## Basic Multimodal Usage

### Simple Image Analysis

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "path/filepath"

    "github.com/inercia/go-llm/pkg/llm"
    "github.com/inercia/go-llm/pkg/factory"
)

func main() {
    // Create client
    factory := factory.New()
    client, err := factory.CreateClient(llm.ClientConfig{
        Provider: "openai",
        Model:    "gpt-4-vision-preview", // Vision-capable model
        APIKey:   "your-api-key",
    })
    if err != nil {
        log.Fatal("Failed to create client:", err)
    }
    defer client.Close()

    // Check vision support
    modelInfo := client.GetModelInfo()
    if !modelInfo.SupportsVision {
        log.Fatal("Model does not support vision")
    }

    // Load image
    imageData, err := os.ReadFile("photo.jpg")
    if err != nil {
        log.Fatal("Failed to read image:", err)
    }

    // Create multimodal request
    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("What do you see in this image? Please describe it in detail."),
                    llm.NewImageContentFromBytes(imageData, "image/jpeg"),
                },
            },
        },
    }

    // Send request
    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        log.Fatal("Request failed:", err)
    }

    fmt.Printf("AI Analysis: %s\n", resp.Choices[0].Message.GetText())
}
```

### Document Analysis

```go
func analyzeDocument(client llm.Client, filePath string) error {
    // Read document
    documentData, err := os.ReadFile(filePath)
    if err != nil {
        return fmt.Errorf("failed to read document: %w", err)
    }

    // Determine MIME type
    mimeType := getMimeTypeFromExtension(filePath)

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Please analyze this document and provide a summary of its key points."),
                    llm.NewFileContentFromBytes(documentData, filepath.Base(filePath), mimeType),
                },
            },
        },
    }

    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return fmt.Errorf("analysis failed: %w", err)
    }

    fmt.Printf("Document Analysis:\n%s\n", resp.Choices[0].Message.GetText())
    return nil
}

func getMimeTypeFromExtension(filename string) string {
    ext := strings.ToLower(filepath.Ext(filename))
    switch ext {
    case ".json":
        return "application/json"
    case ".csv":
        return "text/csv"
    case ".txt":
        return "text/plain"
    case ".xml":
        return "application/xml"
    case ".md":
        return "text/markdown"
    default:
        return "text/plain"
    }
}
```

## Advanced Multimodal Patterns

### Multiple Images Comparison

```go
func compareImages(client llm.Client, image1Path, image2Path string) error {
    // Load both images
    img1Data, err := os.ReadFile(image1Path)
    if err != nil {
        return fmt.Errorf("failed to read first image: %w", err)
    }

    img2Data, err := os.ReadFile(image2Path)
    if err != nil {
        return fmt.Errorf("failed to read second image: %w", err)
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Compare these two images and highlight their differences and similarities:"),
                    llm.NewImageContentFromBytes(img1Data, "image/jpeg"),
                    llm.NewImageContentFromBytes(img2Data, "image/jpeg"),
                },
            },
        },
    }

    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    fmt.Printf("Image Comparison:\n%s\n", resp.Choices[0].Message.GetText())
    return nil
}
```

### Mixed Content Analysis

```go
func analyzeMixedContent(client llm.Client) error {
    // Sample data file
    dataContent := `{
        "sales_data": {
            "Q1": 150000,
            "Q2": 180000,
            "Q3": 165000,
            "Q4": 200000
        },
        "growth_rate": 0.08,
        "regions": ["North", "South", "East", "West"]
    }`

    // Load chart image
    chartImage, err := os.ReadFile("sales_chart.png")
    if err != nil {
        return err
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("I have sales data in JSON format and a corresponding chart. Please analyze both and provide insights:"),
                    llm.NewFileContentFromBytes([]byte(dataContent), "sales_data.json", "application/json"),
                    llm.NewImageContentFromBytes(chartImage, "image/png"),
                    llm.NewTextContent("Do the chart and data match? What trends do you see?"),
                },
            },
        },
    }

    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    fmt.Printf("Mixed Content Analysis:\n%s\n", resp.Choices[0].Message.GetText())
    return nil
}
```

### Sequential Image Processing

```go
func processImageSequence(client llm.Client, imagePaths []string) error {
    var messages []llm.Message

    // Add initial instruction
    messages = append(messages, llm.Message{
        Role: llm.RoleUser,
        Content: []llm.MessageContent{
            llm.NewTextContent("I'll show you a sequence of images. Please analyze them and describe what story they tell together."),
        },
    })

    // Add each image with context
    for i, imagePath := range imagePaths {
        imageData, err := os.ReadFile(imagePath)
        if err != nil {
            return fmt.Errorf("failed to read image %d: %w", i+1, err)
        }

        message := llm.Message{
            Role: llm.RoleUser,
            Content: []llm.MessageContent{
                llm.NewTextContent(fmt.Sprintf("Image %d of %d:", i+1, len(imagePaths))),
                llm.NewImageContentFromBytes(imageData, "image/jpeg"),
            },
        }
        messages = append(messages, message)
    }

    // Add final analysis request
    messages = append(messages, llm.Message{
        Role: llm.RoleUser,
        Content: []llm.MessageContent{
            llm.NewTextContent("Now please analyze the complete sequence and tell me what story these images tell together."),
        },
    })

    req := llm.ChatRequest{Messages: messages}
    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    fmt.Printf("Image Sequence Analysis:\n%s\n", resp.Choices[0].Message.GetText())
    return nil
}
```

## Streaming with Multimodal Content

Multimodal content works with streaming responses:

```go
func streamMultimodalAnalysis(client llm.Client) error {
    imageData, err := os.ReadFile("complex_diagram.png")
    if err != nil {
        return err
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Please provide a detailed analysis of this technical diagram, explaining each component and how they interact:"),
                    llm.NewImageContentFromBytes(imageData, "image/png"),
                },
            },
        },
        // Note: Stream field is set automatically by StreamChatCompletion method
    }

    stream, err := client.StreamChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    fmt.Print("Streaming Analysis: ")
    var fullResponse strings.Builder

    for event := range stream {
        switch {
        case event.IsError():
            return fmt.Errorf("stream error: %s", event.Error.Message)

        case event.IsDelta():
            if len(event.Choice.Delta.Content) > 0 {
                if textContent, ok := event.Choice.Delta.Content[0].(*llm.TextContent); ok {
                    chunk := textContent.GetText()
                    fullResponse.WriteString(chunk)
                    fmt.Print(chunk) // Real-time display
                }
            }

        case event.IsDone():
            fmt.Printf("\n\nAnalysis Complete: %s\n", event.Choice.FinishReason)
            fmt.Printf("Full Analysis: %s\n", fullResponse.String())
        }
    }

    return nil
}
```

## Content Type Utilities

### Image Utilities

```go
// Validate image format
func validateImageFormat(data []byte) error {
    // Check for common image headers
    if len(data) < 4 {
        return fmt.Errorf("invalid image data: too short")
    }

    // JPEG
    if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
        return nil
    }

    // PNG
    if len(data) >= 8 &&
       data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 {
        return nil
    }

    // GIF
    if len(data) >= 6 &&
       string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a" {
        return nil
    }

    // WebP
    if len(data) >= 12 &&
       string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
        return nil
    }

    return fmt.Errorf("unsupported image format")
}

// Resize image if too large
func processImageForLLM(imageData []byte, maxSize int64) ([]byte, error) {
    if int64(len(imageData)) <= maxSize {
        return imageData, nil
    }

    // For production use, implement actual image resizing
    // This is a placeholder that truncates (not recommended)
    log.Printf("Warning: Image too large (%d bytes), consider resizing", len(imageData))
    return imageData[:maxSize], nil
}
```

### File Content Utilities

```go
// Validate text file content
func validateTextFile(data []byte, filename string) error {
    // Check file size limits
    const maxFileSize = 10 * 1024 * 1024 // 10MB
    if len(data) > maxFileSize {
        return fmt.Errorf("file too large: %d bytes (max: %d)", len(data), maxFileSize)
    }

    // Validate UTF-8 encoding
    if !utf8.Valid(data) {
        return fmt.Errorf("file contains invalid UTF-8 encoding")
    }

    // Check for supported file types
    ext := strings.ToLower(filepath.Ext(filename))
    supportedTypes := []string{".txt", ".json", ".csv", ".xml", ".md", ".yaml", ".yml"}

    supported := false
    for _, supportedType := range supportedTypes {
        if ext == supportedType {
            supported = true
            break
        }
    }

    if !supported {
        return fmt.Errorf("unsupported file type: %s", ext)
    }

    return nil
}

// Extract text preview from file
func getFilePreview(data []byte, maxLines int) string {
    lines := strings.Split(string(data), "\n")
    if len(lines) <= maxLines {
        return string(data)
    }

    preview := strings.Join(lines[:maxLines], "\n")
    return preview + fmt.Sprintf("\n... (%d more lines)", len(lines)-maxLines)
}
```

## Best Practices

### 1. Content Size Management

```go
type ContentManager struct {
    maxImageSize int64
    maxFileSize  int64
    maxTotal     int64
}

func NewContentManager() *ContentManager {
    return &ContentManager{
        maxImageSize: 20 * 1024 * 1024, // 20MB per image
        maxFileSize:  10 * 1024 * 1024, // 10MB per file
        maxTotal:     50 * 1024 * 1024, // 50MB total per message
    }
}

func (cm *ContentManager) ValidateMessage(message llm.Message) error {
    var totalSize int64

    for _, content := range message.Content {
        size := content.Size()
        totalSize += size

        switch content.Type() {
        case llm.MessageTypeImage:
            if size > cm.maxImageSize {
                return fmt.Errorf("image too large: %d bytes (max: %d)", size, cm.maxImageSize)
            }
        case llm.MessageTypeFile:
            if size > cm.maxFileSize {
                return fmt.Errorf("file too large: %d bytes (max: %d)", size, cm.maxFileSize)
            }
        }
    }

    if totalSize > cm.maxTotal {
        return fmt.Errorf("message content too large: %d bytes (max: %d)", totalSize, cm.maxTotal)
    }

    return nil
}
```

### 2. Provider Compatibility

```go
func checkMultimodalSupport(client llm.Client) error {
    modelInfo := client.GetModelInfo()

    if !modelInfo.SupportsVision {
        return fmt.Errorf("model does not support vision/image processing")
    }

    // Check specific capabilities if available
    if modelInfo.MaxImageSize > 0 {
        log.Printf("Max image size supported: %d bytes", modelInfo.MaxImageSize)
    }

    return nil
}
```

### 3. Error Handling

```go
func robustMultimodalRequest(client llm.Client, message llm.Message) (*llm.ChatResponse, error) {
    // Validate content before sending
    if err := message.Validate(); err != nil {
        return nil, fmt.Errorf("message validation failed: %w", err)
    }

    req := llm.ChatRequest{Messages: []llm.Message{message}}

    // Retry with exponential backoff for certain errors
    maxRetries := 3
    for i := 0; i < maxRetries; i++ {
        resp, err := client.ChatCompletion(context.Background(), req)
        if err == nil {
            return resp, nil
        }

        // Check if it's a retryable error
        if llmErr, ok := err.(*llm.Error); ok {
            if llmErr.Code == "rate_limit_exceeded" || llmErr.Code == "server_error" {
                backoff := time.Duration(i+1) * time.Second
                log.Printf("Retrying after %v: %v", backoff, err)
                time.Sleep(backoff)
                continue
            }
        }

        return nil, err
    }

    return nil, fmt.Errorf("request failed after %d retries", maxRetries)
}
```

### 4. Content Optimization

```go
// Optimize images before sending
func optimizeImageContent(imageData []byte, mimeType string) ([]byte, error) {
    // Check if image is already reasonably sized
    const targetSize = 5 * 1024 * 1024 // 5MB target
    if len(imageData) <= targetSize {
        return imageData, nil
    }

    // For production, implement actual image compression
    // This is a placeholder
    log.Printf("Image optimization needed: %d bytes -> targeting %d bytes", len(imageData), targetSize)

    // Return original for now - implement compression library in production
    return imageData, nil
}

// Optimize text files
func optimizeFileContent(data []byte, filename string) ([]byte, error) {
    // For large text files, consider chunking or summarization
    const maxSize = 1024 * 1024 // 1MB
    if len(data) <= maxSize {
        return data, nil
    }

    // Truncate with information about truncation
    truncated := data[:maxSize]
    message := fmt.Sprintf("\n\n[Note: File truncated from %d to %d bytes]", len(data), len(truncated))
    return append(truncated, []byte(message)...), nil
}
```

## Common Use Cases

### 1. Document Analysis Workflow

```go
func analyzeBusinessDocument(client llm.Client, docPath string) error {
    docData, err := os.ReadFile(docPath)
    if err != nil {
        return err
    }

    // First, get document summary
    summaryReq := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Please provide a brief summary of this document:"),
                    llm.NewFileContentFromBytes(docData, filepath.Base(docPath), getMimeTypeFromExtension(docPath)),
                },
            },
        },
    }

    summaryResp, err := client.ChatCompletion(context.Background(), summaryReq)
    if err != nil {
        return err
    }

    fmt.Printf("Document Summary:\n%s\n\n", summaryResp.Choices[0].Message.GetText())

    // Then, extract key information
    analysisReq := llm.ChatRequest{
        Messages: []llm.Message{
            summaryReq.Messages[0],
            summaryResp.Choices[0].Message,
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Now extract key data points, important dates, and action items from this document. Format as a structured list."),
                },
            },
        },
    }

    analysisResp, err := client.ChatCompletion(context.Background(), analysisReq)
    if err != nil {
        return err
    }

    fmt.Printf("Key Information:\n%s\n", analysisResp.Choices[0].Message.GetText())
    return nil
}
```

### 2. Visual Content Moderation

```go
func moderateImageContent(client llm.Client, imagePath string) (bool, string, error) {
    imageData, err := os.ReadFile(imagePath)
    if err != nil {
        return false, "", err
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent("Analyze this image for content moderation. Is it appropriate for general audiences? Explain your reasoning."),
                    llm.NewImageContentFromBytes(imageData, "image/jpeg"),
                },
            },
        },
    }

    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return false, "", err
    }

    analysis := resp.Choices[0].Message.GetText()

    // Simple keyword-based decision (enhance with more sophisticated logic)
    inappropriate := strings.Contains(strings.ToLower(analysis), "inappropriate") ||
                    strings.Contains(strings.ToLower(analysis), "not suitable") ||
                    strings.Contains(strings.ToLower(analysis), "offensive")

    return !inappropriate, analysis, nil
}
```

### 3. Educational Content Creation

```go
func createEducationalContent(client llm.Client, imagePath, topic string) error {
    imageData, err := os.ReadFile(imagePath)
    if err != nil {
        return err
    }

    req := llm.ChatRequest{
        Messages: []llm.Message{
            {
                Role: llm.RoleUser,
                Content: []llm.MessageContent{
                    llm.NewTextContent(fmt.Sprintf("Create educational content about %s using this image. Include:\n1. Description of what's shown\n2. Key concepts explained\n3. Fun facts\n4. Discussion questions", topic)),
                    llm.NewImageContentFromBytes(imageData, "image/jpeg"),
                },
            },
        },
    }

    resp, err := client.ChatCompletion(context.Background(), req)
    if err != nil {
        return err
    }

    fmt.Printf("Educational Content for %s:\n%s\n", topic, resp.Choices[0].Message.GetText())
    return nil
}
```

## Troubleshooting

### Common Issues

#### Large Content Handling

```go
// Check content sizes before sending
func validateContentSizes(contents []llm.MessageContent) error {
    for i, content := range contents {
        size := content.Size()

        switch content.Type() {
        case llm.MessageTypeImage:
            if size > 20*1024*1024 { // 20MB limit
                return fmt.Errorf("image %d too large: %d bytes", i, size)
            }
        case llm.MessageTypeFile:
            if size > 10*1024*1024 { // 10MB limit
                return fmt.Errorf("file %d too large: %d bytes", i, size)
            }
        }
    }
    return nil
}
```

#### Provider Limitations

```go
func checkProviderLimits(client llm.Client, message llm.Message) error {
    modelInfo := client.GetModelInfo()

    if !modelInfo.SupportsVision && message.HasContentType(llm.MessageTypeImage) {
        return fmt.Errorf("provider does not support image content")
    }

    if !modelInfo.SupportsFiles && message.HasContentType(llm.MessageTypeFile) {
        return fmt.Errorf("provider does not support file content")
    }

    return nil
}
```

### Performance Tips

1. **Optimize image sizes**: Resize images before sending to reduce processing time
2. **Batch processing**: Send multiple images in one request when possible
3. **Content validation**: Validate content locally before API calls
4. **Caching**: Cache analysis results for identical content
5. **Streaming**: Use streaming for long analyses to improve user experience

## See Also

- [Streaming Documentation](streaming.md) - Streaming multimodal responses
- [Tools Documentation](tools.md) - Combining multimodal content with function calling
- [Architecture Overview](architecture.md) - Understanding content processing architecture
- [Examples Directory](../examples/) - Complete multimodal examples
