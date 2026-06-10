# Gollek Tool Core Module

This module contains the core tool-related functionality for the Gollek inference engine, implementing the features required for robust tool calling as specified in the enhancement document.

## Features Implemented

### ✅ Tool Registry
- Interface: `tech.kayys.gollek.spi.registry.ToolRegistry`
- Implementation: Available through dependency injection
- Provides registration, lookup, and listing of available tools

### ✅ JSON Schema for Tools
- Tools define their input parameters using JSON Schema
- Used for validation and documentation purposes
- Integrated with the Tool interface via `inputSchema()` method

### ✅ Tool Call Detection
- Implemented in the reasoning plugin (`gollek-plugin-reasoning`)
- Detects tool calls in LLM output using multiple formats (JSON, XML-style)
- Integrates with the inference pipeline

### ✅ Argument Validation
- Validator: `tech.kayys.gollek.tool.validation.ToolArgumentValidator`
- Validates arguments against the tool's JSON schema
- Checks for required parameters and type correctness
- Throws `ToolValidationException` on validation failure

### ✅ Tool Execution
- Executor: `tech.kayys.gollek.tool.impl.DefaultToolExecutor`
- Executes tools after validating arguments
- Handles success and failure cases
- Returns structured execution results

### ✅ Tool Result Injection
- Formatter: `tech.kayys.gollek.tool.util.ToolResultFormatter`
- Formats tool execution results for LLM consumption
- Injects results back into conversation history
- Enables multi-turn reasoning with tool results

### ✅ Multiple Tool Calls Per Turn (Future)
- Architecture supports processing multiple tool calls in a single turn
- Sequential execution with error handling
- Ready for parallel execution optimization

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Tool Call     │───▶│  Tool Execution  │───▶│  Result         │
│   Detection     │    │  Plugin          │    │  Injection      │
│ (Reasoning)     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────────┐
                       │  Tool Registry   │
                       │  & Validation    │
                       └──────────────────┘
                              │
                       ┌──────────────────┐
                       │  Tool Execution  │
                       │  Engine          │
                       └──────────────────┘
```

## Key Classes

- `ToolExecutionResult`: Standardized result format with success/failure states
- `ToolArgumentValidator`: Validates inputs against JSON schema
- `ToolResultFormatter`: Prepares results for LLM consumption
- `InvocationStatus`: Enum for tracking execution status

## Integration Points

The module integrates with:
- Gollek SPI for core interfaces
- Tool registry for tool discovery
- Inference pipeline for execution
- Conversation history for result injection
