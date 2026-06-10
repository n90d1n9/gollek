package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentStreamEventParserTest {
    private final AgentStreamEventParser parser = new AgentStreamEventParser(new ObjectMapper());

    @Test
    void parsesChatCompletionChunk() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "id": "chatcmpl-req-1",
                  "object": "chat.completion.chunk",
                  "choices": [
                    {
                      "index": 0,
                      "delta": {"role": "assistant", "content": "hello"},
                      "finish_reason": null
                    }
                  ],
                  "metadata": {
                    "gollek_trace": {"request_id": "req-1", "trace_id": "trace-1"},
                    "gollek_stream": {
                      "surface": "chat.completions",
                      "sequence_number": 0,
                      "final": false,
                      "include_usage": true
                    }
                  }
                }
                """);

        assertEquals(AgentStreamEvent.Surface.CHAT_COMPLETIONS, event.surface());
        assertEquals("chat.completion.chunk", event.type());
        assertEquals(0, event.sequenceNumber());
        assertEquals("chatcmpl-req-1", event.id());
        assertEquals("assistant", event.role());
        assertEquals("hello", event.delta());
        assertEquals("req-1", event.trace().get("request_id"));
        assertTrue(event.hasDelta());
    }

    @Test
    void parsesChatFinalUsageChunk() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "id": "chatcmpl-req-1",
                  "object": "chat.completion.chunk",
                  "choices": [
                    {
                      "index": 0,
                      "delta": {},
                      "finish_reason": "stop"
                    }
                  ],
                  "metadata": {
                    "gollek_stream": {
                      "surface": "chat.completions",
                      "sequence_number": 1,
                      "final": true,
                      "finish_reason": "stop"
                    }
                  },
                  "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5
                  }
                }
                """);

        assertEquals("stop", event.finishReason());
        assertEquals(1, event.sequenceNumber());
        assertEquals(2, event.usage().promptTokens());
        assertEquals(3, event.usage().completionTokens());
        assertEquals(5, event.usage().totalTokens());
        assertTrue(event.isCompleted());
    }

    @Test
    void exposesChatToolCallDelta() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "id": "chatcmpl-req-1",
                  "object": "chat.completion.chunk",
                  "choices": [
                    {
                      "index": 0,
                      "delta": {
                        "tool_calls": [
                          {
                            "index": 0,
                            "id": "call_search",
                            "type": "function",
                            "function": {
                              "name": "knowledge_search",
                              "arguments": "{\\"query\\":\\"gollek\\"}"
                            }
                          }
                        ]
                      },
                      "finish_reason": null
                    }
                  ]
                }
                """);

        assertTrue(event.hasToolCalls());
        assertEquals(1, event.toolCalls().size());
        AgentStreamEvent.ToolCall call = event.toolCalls().get(0);
        assertEquals("call_search", call.id());
        assertEquals(0, call.index());
        assertEquals("function", call.type());
        assertEquals("knowledge_search", call.name());
        assertEquals("{\"query\":\"gollek\"}", call.arguments());
        assertEquals("gollek", call.argumentsJson().path("query").asText());
        assertEquals(Map.of("query", "gollek"), call.argumentsMap());
    }

    @Test
    void parsesResponsesCompletedEvent() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "type": "response.completed",
                  "sequence_number": 3,
                  "response": {
                    "id": "resp_req_1",
                    "status": "completed",
                    "output_text": "done",
                    "metadata": {
                      "gollek_trace": {"request_id": "req-1"},
                      "gollek_stream": {
                        "surface": "responses",
                        "sequence_number": 3,
                        "final": true,
                        "include_usage": true,
                        "finish_reason": "stop"
                      }
                    },
                    "usage": {
                      "prompt_tokens": 4,
                      "completion_tokens": 5,
                      "total_tokens": 9
                    }
                  }
                }
                """);

        assertEquals(AgentStreamEvent.Surface.RESPONSES, event.surface());
        assertEquals("response.completed", event.type());
        assertEquals(3, event.sequenceNumber());
        assertEquals("resp_req_1", event.responseId());
        assertEquals("done", event.outputText());
        assertEquals("stop", event.finishReason());
        assertEquals(9, event.usage().totalTokens());
        assertEquals("req-1", event.trace().get("request_id"));
        assertTrue(event.isCompleted());
    }

    @Test
    void exposesResponsesFunctionCallArgumentDelta() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "type": "response.function_call_arguments.delta",
                  "sequence_number": 4,
                  "response_id": "resp_req_1",
                  "item_id": "fc_search",
                  "output_index": 1,
                  "delta": "{\\"query\\":\\"gollek",
                  "call_id": "call_search"
                }
                """);

        assertTrue(event.hasToolCalls());
        AgentStreamEvent.ToolCall call = event.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals(1, call.index());
        assertEquals("response.function_call_arguments.delta", call.type());
        assertEquals("call_search", call.callId());
        assertEquals("{\"query\":\"gollek", call.arguments());
    }

    @Test
    void exposesResponsesCompletedFunctionCallOutputItem() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "type": "response.completed",
                  "sequence_number": 3,
                  "response": {
                    "id": "resp_req_1",
                    "status": "completed",
                    "output": [
                      {
                        "id": "msg_req_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "I will search."}]
                      },
                      {
                        "id": "fc_search",
                        "type": "function_call",
                        "name": "knowledge_search",
                        "arguments": "{\\"query\\":\\"gollek\\"}",
                        "call_id": "call_search",
                        "status": "completed"
                      }
                    ]
                  }
                }
                """);

        assertTrue(event.hasToolCalls());
        AgentStreamEvent.ToolCall call = event.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals("function_call", call.type());
        assertEquals("knowledge_search", call.name());
        assertEquals("call_search", call.callId());
        assertEquals("completed", call.status());
    }

    @Test
    void parsesResponsesErrorEvent() throws Exception {
        AgentStreamEvent event = parser.parse("""
                {
                  "type": "error",
                  "sequence_number": 0,
                  "error": {
                    "message": "Streaming response failed",
                    "type": "server_error"
                  }
                }
                """);

        assertEquals(AgentStreamEvent.Surface.RESPONSES, event.surface());
        assertEquals("error", event.type());
        assertEquals("Streaming response failed", event.errorMessage());
        assertTrue(event.isError());
    }
}
