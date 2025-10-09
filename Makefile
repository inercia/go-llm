.PHONY: all build test lint format clean help deps security check-deps pre-commit test-coverage test-sequential test-max-parallel test-medium-parallel forat

# Variables
BINARY_NAME=go-llm
GO_FILES=$(shell find . -name '*.go' -type f | grep -v vendor/)
MD_FILES=$(shell find . -name '*.md' -type f | grep -v vendor/ | grep -v node_modules/)
PACKAGES=$(shell go list ./...)
GO_MODULE=$(shell sed -n 's/^module //p' go.mod)
GO_BIN=$(shell go env GOBIN)

# Default target
all: format lint test build

# Build the project
build:
	@echo "Building..."
	go build -v ./...

# Run tests
test:
	@echo "Running tests..."
	go test -v -race -parallel 4 -coverprofile=coverage.out ./pkg/...

# Run tests with coverage
test-coverage: test
	@echo "Generating coverage report..."
	go tool cover -html=coverage.out -o coverage.html

# Run tests sequentially (no parallel)
test-sequential:
	@echo "Running tests sequentially..."
	go test -v -race -parallel 1 -coverprofile=coverage.out ./pkg/...

# Run tests with maximum parallelization
test-max-parallel:
	@echo "Running tests with maximum parallelization..."
	go test -v -race -parallel 16 -coverprofile=coverage.out ./pkg/...

# Run tests with medium parallelization
test-medium-parallel:
	@echo "Running tests with medium parallelization..."
	go test -v -race -parallel 8 -coverprofile=coverage.out ./pkg/...

test-integration:
	@echo "Running integration tests..."
	@echo "... related env vars:"
	@env | grep -E "(AWS_|BEDROCK_|GEMINI_|OPENAI_|OPENROUTER_|DEEPSEEK_)"
	go test -v -timeout=300s -parallel 4 -race ./test/...

# Provider-specific tests
test-openai:
	@echo "Running OpenAI tests..."
	go test -v -race ./pkg/llm/ -run 'OpenAI'

test-openrouter:
	@echo "Running OpenRouter tests..."
	go test -v -race ./pkg/llm/ -run 'OpenRouter'

test-gemini:
	@echo "Running Gemini tests..."
	go test -v -race ./pkg/llm/ -run 'Gemini'

test-ollama:
	@echo "Running Ollama tests..."
	go test -v -race ./pkg/llm/ -run 'Ollama'

test-deepseek:
	@echo "Running DeepSeek tests..."
	go test -v -race ./pkg/llm/ -run 'DeepSeek'

# Run linting
lint:
	@echo "Running linter..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not found. Installing..."; \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.4.0; \
		golangci-lint run; \
	fi

# Format code
format:
	@echo "Formatting code (goimports)..."
	@if command -v goimports >/dev/null 2>&1; then \
		goimports -w -local $(GO_MODULE) $(GO_FILES); \
	else \
		echo "goimports not found. Installing..."; \
		go install golang.org/x/tools/cmd/goimports@latest; \
		goimports -w -local $(GO_MODULE) $(GO_FILES); \
	fi
	go mod tidy
	@echo "Formatting markdown files..."
	@if command -v prettier >/dev/null 2>&1; then \
		prettier --write $(MD_FILES); \
	elif command -v npm >/dev/null 2>&1; then \
		echo "prettier not found. Installing..."; \
		npm install -g prettier@latest; \
		prettier --write $(MD_FILES); \
	else \
		echo "Warning: npm not found. Please install Node.js and npm to format markdown files."; \
		echo "Skipping markdown formatting..."; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f coverage.out coverage.html
	go clean ./...
	find . -name '*.log' -type f -delete

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod verify

# Security check
security:
	@echo "Running security check..."
	@if command -v gosec >/dev/null 2>&1; then \
		gosec ./...; \
	else \
		echo "gosec not found. Installing..."; \
		go install github.com/securego/gosec/v2/cmd/gosec@latest; \
		gosec ./...; \
	fi

# Check for outdated dependencies
check-deps:
	@echo "Checking for outdated dependencies..."
	go list -u -m all

# Run pre-commit checks
pre-commit: format lint test

# Help
help:
	@echo "Available targets:"
	@echo "  all                 - Run format, lint, test, and build"
	@echo "  build               - Build the project"
	@echo "  test                - Run tests with race detection (parallel=4)"
	@echo "  test-sequential     - Run tests sequentially (no parallelization)"
	@echo "  test-medium-parallel- Run tests with medium parallelization (parallel=8)"
	@echo "  test-max-parallel   - Run tests with maximum parallelization (parallel=16)"
	@echo "  test-coverage       - Run tests and generate coverage report"
	@echo "  lint                - Run golangci-lint"
	@echo "  format              - Format Go code with goimports (-local $(GO_MODULE)), tidy modules, and format markdown files"
	@echo "  clean               - Clean build artifacts"
	@echo "  deps                - Install and verify dependencies"
	@echo "  security            - Run security analysis with gosec"
	@echo "  check-deps          - Check for outdated dependencies"
	@echo "  pre-commit          - Run format, lint, and test (useful for pre-commit hooks)"
	@echo "  help                - Show this help message"
	@echo "  forat               - Alias for 'format' (typo)"
	@echo "  test-openai        - Run OpenAI specific tests"
	@echo "  test-openrouter    - Run OpenRouter specific tests"
	@echo "  test-gemini        - Run Gemini specific tests"
	@echo "  test-ollama        - Run Ollama specific tests"
	@echo "  test-deepseek      - Run DeepSeek specific tests"

# Convenience alias for a common typo
forat: format
