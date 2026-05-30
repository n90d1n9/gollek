package tech.kayys.gollek.server.api.v1;

final class AgentOpenApiExamples {

    static final String CHAT_REQUEST = """
            {
              "model": "demo-model",
              "request_id": "req_agent_123",
              "trace_id": "trace_agent_123",
              "messages": [
                {"role": "system", "content": "Answer using supplied context and tools when relevant."},
                {"role": "user", "content": "What changed for agentic support?"}
              ],
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "search_context",
                    "description": "Search external knowledge",
                    "parameters": {
                      "type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]
                    }
                  },
                  "x_gollek": {"mcp_server": "knowledge"}
                }
              ],
              "tool_choice": "auto",
              "rag_context": [
                {
                  "id": "chunk-1",
                  "source": "docs/agentic",
                  "score": 0.92,
                  "text": "Gollek exposes inference APIs for agent orchestrators."
                }
              ]
            }
            """;

    static final String CHAT_STREAM_REQUEST = """
            {
              "model": "demo-model",
              "request_id": "req_agent_stream_123",
              "trace_id": "trace_agent_stream_123",
              "stream": true,
              "stream_options": {
                "include_usage": true,
                "include_trace": true,
                "include_stream_metadata": true
              },
              "messages": [
                {"role": "system", "content": "Answer using supplied context when relevant."},
                {"role": "user", "content": "Stream the agent-facing answer."}
              ],
              "rag_context": [
                {
                  "source": "docs/agentic",
                  "text": "Gollek streams inference events and leaves orchestration to callers."
                }
              ]
            }
            """;

    static final String CHAT_RESPONSE = """
            {
              "id": "chatcmpl-request-id",
              "object": "chat.completion",
              "created": 1760000000,
              "model": "demo-model",
              "choices": [
                {
                  "index": 0,
                  "message": {"role": "assistant", "content": "Gollek provides serving APIs for agent projects."},
                  "finish_reason": "stop"
                }
              ],
              "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
              "metadata": {
                "gollek_trace": {
                  "request_id": "req_agent_123",
                  "trace_id": "trace_agent_123"
                }
              }
            }
            """;

    static final String CHAT_STREAM = """
            data:{"id":"chatcmpl-request-id","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hello"}}],"metadata":{"gollek_trace":{"request_id":"req_agent_stream_123","trace_id":"trace_agent_stream_123"},"gollek_stream":{"surface":"chat.completions","sequence_number":0,"final":false,"include_usage":true}}}

            data:{"id":"chatcmpl-request-id","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"metadata":{"gollek_stream":{"surface":"chat.completions","sequence_number":1,"final":true,"include_usage":true,"finish_reason":"stop"}},"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}

            data:[DONE]
            """;

    static final String RESPONSES_REQUEST = """
            {
              "model": "demo-model",
              "instructions": "Use the provided context when relevant.",
              "input": "Summarize Gollek's agent integration boundary.",
              "retrieval_context": {
                "documents": [
                  {
                    "content": "Gollek owns serving, MCP discovery, embeddings, and RAG context injection.",
                    "metadata": {"source": "docs/api"}
                  }
                ]
              },
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"}
                  }
                }
              ]
            }
            """;

    static final String RESPONSES_STREAM_REQUEST = """
            {
              "model": "demo-model",
              "request_id": "req_response_stream_123",
              "trace_id": "trace_response_stream_123",
              "stream": true,
              "stream_options": {
                "include_usage": true,
                "include_trace": true,
                "include_stream_metadata": true
              },
              "instructions": "Use the provided context when relevant.",
              "input": "Stream Gollek's agent integration boundary.",
              "retrieval_context": [
                {
                  "source": "docs/api",
                  "text": "Gollek owns serving, MCP discovery, embeddings, and RAG context injection."
                }
              ]
            }
            """;

    static final String RESPONSES_RESPONSE = """
            {
              "id": "resp_request-id",
              "object": "response",
              "status": "completed",
              "model": "demo-model",
              "output": [
                {
                  "id": "msg_request-id",
                  "type": "message",
                  "role": "assistant",
                  "content": [{"type": "output_text", "text": "Gollek is the serving boundary."}]
                }
              ],
              "output_text": "Gollek is the serving boundary."
            }
            """;

    static final String RESPONSES_STREAM = """
            data:{"type":"response.created","sequence_number":0,"response":{"id":"resp_request-id","status":"in_progress","metadata":{"gollek_trace":{"request_id":"req_response_stream_123","trace_id":"trace_response_stream_123"},"gollek_stream":{"surface":"responses","sequence_number":0,"final":false,"include_usage":true}}}}

            data:{"type":"response.output_text.delta","sequence_number":1,"delta":"Hello","metadata":{"gollek_stream":{"surface":"responses","sequence_number":1,"final":false,"include_usage":true}}}

            data:{"type":"response.completed","sequence_number":3,"response":{"id":"resp_request-id","status":"completed","output_text":"Hello","metadata":{"gollek_stream":{"surface":"responses","sequence_number":3,"final":true,"include_usage":true,"finish_reason":"stop"}},"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}}

            data:[DONE]
            """;

    static final String EMBEDDINGS_REQUEST = """
            {
              "model": "demo-embed",
              "request_id": "req_embed_123",
              "trace_id": "trace_embed_123",
              "input": ["Gollek serves embeddings for RAG pipelines."],
              "metadata": {"tenant": "agent-project"}
            }
            """;

    static final String EMBEDDINGS_RESPONSE = """
            {
              "object": "list",
              "model": "demo-embed",
              "data": [
                {"object": "embedding", "index": 0, "embedding": [0.0, 0.0, 0.0]}
              ],
              "metadata": {
                "gollek_trace": {
                  "request_id": "req_embed_123",
                  "trace_id": "trace_embed_123"
                }
              }
            }
            """;

    static final String ERROR_RESPONSE = """
            {
              "error": {
                "message": "model is required",
                "type": "invalid_request_error",
                "request_id": "req_agent_123",
                "trace_id": "trace_agent_123"
              }
            }
            """;

    static final String VALIDATION_REQUEST = """
            {
              "model": "demo-model",
              "request_id": "req_validate_123",
              "trace_id": "trace_validate_123",
              "messages": [
                {"role": "system", "content": "Use supplied context."},
                {"role": "user", "content": "Validate this agent request."}
              ],
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"}
                  },
                  "x_gollek": {"mcp_server": "knowledge"}
                }
              ],
              "rag_context": [
                {"source": "docs/agentic", "text": "Validation does not invoke the model."}
              ]
            }
            """;

    static final String VALIDATION_RESPONSE = """
            {
              "object": "gollek.agent_validation",
              "surface": "chat",
              "valid": true,
              "model_invoked": false,
              "trace": {
                "request_id": "req_validate_123",
                "trace_id": "trace_validate_123"
              },
              "normalized": {
                "request_id": "req_validate_123",
                "trace_id": "trace_validate_123",
                "model": "demo-model",
                "message_count": 3,
                "tool_count": 1,
                "tool_contract": {"valid": true, "tool_count": 1, "warning_count": 0},
                "rag": {"injected": true, "items": 1}
              },
              "boundary": {
                "validation_only": true,
                "tool_execution": false,
                "retrieval_execution": false
              }
            }
            """;

    static final String TOOL_VALIDATION_REQUEST = """
            {
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "lookup_context",
                    "strict": true,
                    "parameters": {
                      "type": "object",
                      "properties": {
                        "query": {
                          "type": "string",
                          "anyOf": [{"minLength": 1}]
                        }
                      },
                      "required": ["query"]
                    }
                  },
                  "x_gollek": {"mcp_server": "knowledge"}
                }
              ]
            }
            """;

    static final String TOOL_VALIDATION_RESPONSE = """
            {
              "object": "gollek.tool_contract_validation",
              "valid": true,
              "model_invoked": false,
              "tool_count": 1,
              "normalized": [
                {
                  "name": "lookup_context",
                  "type": "function",
                  "strict": true,
                  "parameter_keys": ["properties", "required", "type"],
                  "metadata": {"mcp_server": "knowledge"}
                }
              ],
              "warnings": [
                {"code": "schema_feature_may_be_ignored"}
              ],
              "boundary": {
                "validation_only": true,
                "tool_execution": false,
                "tool_authorization": false
              }
            }
            """;

    static final String AGENT_CONTRACT_RESPONSE = """
            {
              "object": "gollek.agent_contract",
              "version": "v1",
              "service_role": "inference_serving_engine",
              "boundary": {
                "tool_execution": false,
                "retrieval_execution": false
              },
              "endpoints": {
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"}
              }
            }
            """;

    static final String MCP_SERVERS_RESPONSE = """
            {
              "available": true,
              "registry_path": "/home/user/.gollek/mcp.json",
              "servers": [
                {"name": "knowledge", "enabled": true}
              ],
              "boundary": {"role": "discovery_only", "tool_execution": false}
            }
            """;

    static final String MCP_TOOLS_RESPONSE = """
            {
              "available": true,
              "compat": "openai",
              "enabled_only": true,
              "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "mcp_knowledge_search",
                    "description": "Search knowledge",
                    "parameters": {"type": "object"}
                  },
                  "x_gollek": {
                    "mcp_server": "knowledge",
                    "mcp_tool_name": "search",
                    "tool_execution": false
                  }
                }
              ]
            }
            """;

    private AgentOpenApiExamples() {
    }
}
