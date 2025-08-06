# Multi-Model Benchmarking Usage Guide

This document explains how to use the multi-model benchmarking functionality with YAML configuration files.

## Overview

The `multi_benchmark.py` script allows you to benchmark multiple models from different providers with individual configurations for each model. It uses YAML configuration files for maximum flexibility and readability.

## Quick Start

### 1. Create Configuration File

Generate an example configuration file:
```bash
cd /home/michal/projects/hue/prometheus_ai/prometheus_ai
uv run python multi_benchmark.py --create-example my_config.yaml
```

### 2. Edit Configuration

Edit `my_config.yaml` to match your models and settings:
```yaml
global:
  seed: 42
  samples: 50
  skip_actions:
    - set_color
    - dim

models:
  - model: gpt-4
    provider: openai
    max_concurrent_requests: 3
    max_retries: 2

  - model: claude-3-sonnet
    provider: anthropic
    max_concurrent_requests: 5
    max_retries: 1

  - model: local-llama
    provider: local
    max_concurrent_requests: 2
    max_retries: 3
```

### 3. Run Benchmark

```bash
uv run python multi_benchmark.py my_config.yaml
```

## Configuration Structure

### Global Settings

All models share these settings:
- `seed`: Random seed for reproducible results (default: 42)
- `samples`: Number of scenarios to test (default: 10)
- `skip_actions`: Actions to exclude from testing (default: ["set_color", "dim"])

### Model-Specific Settings

Each model requires these fields:
- `model`: Model name (**required**)
- `provider`: Provider (**required** - e.g., "openai", "anthropic", "mistral")

Optional fields with sensible defaults:
- `max_concurrent_requests`: Concurrent requests for this model (default: 5)
- `max_retries`: Maximum retries for failed requests (default: 1)

## Comprehensive YAML Configuration Example

Here's a complete example showing all configuration options and real-world usage patterns:

```yaml
# Multi-Model Benchmark Configuration
# This example shows comprehensive configuration for comparing multiple AI models

global:
  # Reproducibility settings - SAME for all models to ensure fair comparison
  seed: 42                    # Fixed seed ensures identical test scenarios
  samples: 100                # Number of test scenarios to run per model
  
  # Actions to skip during testing (optional)
  skip_actions:
    - set_color               # Skip color-setting commands
    - dim                     # Skip dimming commands
    - turn_on                 # Skip turn-on commands (if problematic)

# Model configurations - each can have different performance settings
models:
  # OpenAI Models
  - model: gpt-4o             # Latest GPT-4 model
    provider: openai
    max_concurrent_requests: 8  # Higher concurrency for fast API
    max_retries: 2
    
  - model: gpt-4o-mini        # Cost-effective option
    provider: openai
    max_concurrent_requests: 12 # Even higher for cheaper model
    max_retries: 1
    
  - model: gpt-4-turbo        # Previous generation
    provider: openai
    max_concurrent_requests: 6
    max_retries: 2

  # Anthropic Models  
  - model: claude-3-5-sonnet-20241022  # Latest Claude model
    provider: anthropic
    max_concurrent_requests: 5  # Conservative for Claude API
    max_retries: 3              # More retries for reliability
    
  - model: claude-3-haiku-20240307     # Fast, lightweight model
    provider: anthropic
    max_concurrent_requests: 10
    max_retries: 1

  # Google Models
  - model: gemini-1.5-pro     # Google's flagship model
    provider: google
    max_concurrent_requests: 4
    max_retries: 2
    
  - model: gemini-1.5-flash   # Faster, cheaper option
    provider: google
    max_concurrent_requests: 8
    max_retries: 1

  # Open Source / Self-Hosted Models
  - model: llama-3.1-70b-instruct  # Meta's Llama model
    provider: ollama                 # Local hosting via Ollama
    max_concurrent_requests: 2       # Lower for local resources
    max_retries: 5                   # More retries for local stability
    
  - model: mixtral-8x7b-instruct    # Mistral's mixture of experts
    provider: together               # Hosted on Together AI
    max_concurrent_requests: 6
    max_retries: 2

  # Specialized Models
  - model: claude-3-opus-20240229     # Highest quality for comparison
    provider: anthropic
    max_concurrent_requests: 2        # Very conservative for expensive model
    max_retries: 3
    
  # Minimal configuration examples (uses defaults)
  - model: gpt-3.5-turbo      # Only required fields specified
    provider: openai          # Gets defaults: max_concurrent_requests=5, max_retries=1
    
  - model: claude-3-sonnet-20240229
    provider: anthropic       # Uses all default performance settings
```

### Configuration Validation Notes

**Required Fields:**
- Every model MUST have `model` and `provider` fields
- The `global` section is optional (uses defaults if missing)
- The `models` array must contain at least one model

**Default Behavior:**
- If you omit `max_concurrent_requests`, it defaults to 5
- If you omit `max_retries`, it defaults to 1  
- If you omit `global` settings, sensible defaults are used
- User-specified values always override defaults

**Global Consistency:**
- All models use the SAME `seed`, `samples`, and `skip_actions`
- This ensures fair comparison with identical test scenarios
- Different models cannot have different seeds (prevents unfair comparisons)

## Advanced Examples

### Different Providers and Settings
```yaml
global:
  seed: 123
  samples: 100
  skip_actions:
    - set_color
    - dim
    - turn_on

models:
  - model: gpt-4-turbo
    provider: openai
    max_concurrent_requests: 5
    max_retries: 2

  - model: claude-3-opus
    provider: anthropic
    max_concurrent_requests: 3
    max_retries: 3

  - model: mistral-large
    provider: mistral
    max_concurrent_requests: 4
    max_retries: 1

  - model: llama-3.1-405b
    provider: local
    max_concurrent_requests: 1
    max_retries: 5
```

### High-Performance Testing
```yaml
global:
  seed: 42
  samples: 1000
  skip_actions: []

models:
  - model: gpt-4o-mini
    provider: openai
    max_concurrent_requests: 10
    max_retries: 1

  - model: claude-3-haiku
    provider: anthropic
    max_concurrent_requests: 15
    max_retries: 1
```

### Comments and Documentation
YAML supports comments, making configurations self-documenting:
```yaml
# Multi-model benchmark configuration
# Purpose: Compare OpenAI vs Anthropic models on hue commands

global:
  seed: 42              # Reproducible results
  samples: 100          # Comprehensive test
  skip_actions:
    - set_color         # Skip problematic actions
    - dim

models:
  # Fast, cost-effective model
  - model: gpt-4o-mini
    provider: openai
    max_concurrent_requests: 8
    max_retries: 1

  # High-quality model
  - model: claude-3-sonnet
    provider: anthropic
    max_concurrent_requests: 5
    max_retries: 2
```

## Output Structure

### Console Output
- Real-time progress for each model
- Configuration summary
- Rich table with results comparison
- File locations

### Generated Files
```
benchmarking/logs/
├── multi_benchmark_results_3_models_150_20250805_140000.jsonl  # All results
└── multi_benchmark_summary_20250805_140000.json                # Summary
```

### JSONL File Format
```jsonl
{"metadata": {"config_file": "...", "models": [...], ...}}
{"multi_benchmark_results": {"total_models": 3, "model_summaries": [...]}}
{"scenario": {...}, "model": "gpt-4", "provider": "openai", ...}
{"scenario": {...}, "model": "claude-3", "provider": "anthropic", ...}
...
```

## Benefits

### Flexibility
- Different concurrent request limits per model
- Individual retry strategies
- Provider-specific optimizations
- Easy configuration management

### Readability
- YAML is human-readable and easy to edit
- Support for comments and documentation
- Clean, hierarchical structure
- Version control friendly

### Reproducibility
- Same seed ensures identical scenarios
- Configuration file preserves exact settings
- Complete run metadata saved

### Scalability
- Add/remove models easily
- Configure per-model performance settings
- Handle provider-specific limitations

## Usage Patterns

### Development Testing
```bash
# Quick test with few samples
uv run python multi_benchmark.py --create-example dev_config.yaml
# Edit dev_config.yaml to set samples: 5
uv run python multi_benchmark.py dev_config.yaml
```

### Production Benchmarking
```bash
# Full evaluation
uv run python multi_benchmark.py production_config.yaml
```

### A/B Testing
```bash
# Compare specific models
uv run python multi_benchmark.py ab_test_config.yaml
```

## Error Handling

- Invalid configuration files show clear error messages
- Individual model failures don't stop the entire run
- Network timeouts are retried per model settings
- All errors logged to both console and logfire

## Integration with Existing Workflow

- Uses the same `benchmark()` function from `benchmarking.py`
- Same scoring logic and trajectory format
- Compatible with existing log analysis tools
- Maintains logfire integration patterns

This YAML-based approach provides maximum flexibility and readability while maintaining consistency with the existing benchmarking infrastructure.
