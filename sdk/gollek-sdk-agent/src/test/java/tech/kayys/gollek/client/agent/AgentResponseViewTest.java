package tech.kayys.gollek.client.agent;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentResponseViewTest {

    @Test
    void extractsChatCompletionTextUsageAndToolCalls() throws Exception {
        AgentResponseView view = AgentResponseView.fromJson("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion",
                  "model": "demo",
                  "choices": [
                    {
                      "index": 0,
                      "message": {
                        "role": "assistant",
                        "content": "I will search.",
                        "tool_calls": [
                          {
                            "id": "call_search",
                            "type": "function",
                            "function": {
                              "name": "knowledge_search",
                              "arguments": "{\\"query\\":\\"gollek\\"}"
                            }
                          }
                        ]
                      },
                      "finish_reason": "tool_calls"
                    }
                  ],
                  "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9
                  }
                }
                """);

        assertEquals(AgentStreamEvent.Surface.CHAT_COMPLETIONS, view.surface());
        assertEquals("chatcmpl-1", view.id());
        assertEquals("demo", view.model());
        assertEquals("I will search.", view.outputText());
        assertEquals("tool_calls", view.finishReason());
        assertEquals(9, view.usage().totalTokens());
        assertTrue(view.hasToolCalls());

        AgentStreamEvent.ToolCall call = view.toolCalls().get(0);
        assertEquals("call_search", call.id());
        assertEquals("function", call.type());
        assertEquals("knowledge_search", call.name());
        assertEquals(Map.of("query", "gollek"), call.argumentsMap());
    }

    @Test
    void extractsResponsesOutputTraceAndToolCalls() throws Exception {
        AgentResponseView view = AgentResponseView.fromJson("""
                {
                  "id": "resp_1",
                  "object": "response",
                  "model": "demo",
                  "status": "completed",
                  "output": [
                    {
                      "id": "msg_1",
                      "type": "message",
                      "status": "completed",
                      "role": "assistant",
                      "content": [
                        {"type": "output_text", "text": "I will search.", "annotations": []}
                      ]
                    },
                    {
                      "id": "fc_search",
                      "type": "function_call",
                      "name": "knowledge_search",
                      "arguments": "{\\"query\\":\\"gollek\\"}",
                      "call_id": "call_search",
                      "status": "completed"
                    }
                  ],
                  "output_text": "I will search.",
                  "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 5,
                    "total_tokens": 9
                  },
                  "metadata": {
                    "gollek_trace": {
                      "request_id": "req-1",
                      "trace_id": "trace-1"
                    }
                  }
                }
                """);

        assertEquals(AgentStreamEvent.Surface.RESPONSES, view.surface());
        assertEquals("resp_1", view.id());
        assertEquals("I will search.", view.outputText());
        assertEquals("req-1", view.trace().get("request_id"));
        assertEquals(9, view.usage().totalTokens());
        assertTrue(view.hasToolCalls());

        AgentStreamEvent.ToolCall call = view.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals("function_call", call.type());
        assertEquals("call_search", call.callId());
        assertEquals("completed", call.status());
        assertEquals(Map.of("query", "gollek"), call.argumentsMap());
    }

    @Test
    void extractsResponsesTextFromOutputContentWhenTopLevelTextIsMissing() throws Exception {
        AgentResponseView view = AgentResponseView.fromJson("""
                {
                  "id": "resp_1",
                  "object": "response",
                  "output": [
                    {
                      "type": "message",
                      "content": [
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": " world"}
                      ]
                    }
                  ]
                }
                """);

        assertEquals("Hello world", view.outputText());
    }
}
