package tech.kayys.gollek.client.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AgentStreamAccumulatorTest {
    private final AgentStreamEventParser parser = new AgentStreamEventParser(new ObjectMapper());

    @Test
    void accumulatesTextUsageAndCompletion() throws Exception {
        AgentStreamAccumulator accumulator = new AgentStreamAccumulator();

        accumulator.accept(parser.parse("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion.chunk",
                  "choices": [{"delta": {"role": "assistant", "content": "Hel"}, "finish_reason": null}]
                }
                """));
        accumulator.accept(parser.parse("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion.chunk",
                  "choices": [{"delta": {"content": "lo"}, "finish_reason": null}]
                }
                """));
        AgentStreamAccumulator.Snapshot snapshot = accumulator.accept(parser.parse("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion.chunk",
                  "choices": [{"delta": {}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
                }
                """));

        assertEquals("Hello", snapshot.outputText());
        assertEquals("stop", snapshot.finishReason());
        assertEquals(3, snapshot.usage().totalTokens());
        assertTrue(snapshot.completed());
        assertFalse(snapshot.hasToolCalls());
    }

    @Test
    void mergesFragmentedChatToolCallArgumentsByIndex() throws Exception {
        AgentStreamAccumulator accumulator = new AgentStreamAccumulator();

        accumulator.accept(parser.parse("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion.chunk",
                  "choices": [
                    {
                      "delta": {
                        "tool_calls": [
                          {
                            "index": 0,
                            "id": "call_search",
                            "type": "function",
                            "function": {
                              "name": "knowledge_search",
                              "arguments": "{\\"query\\""
                            }
                          }
                        ]
                      },
                      "finish_reason": null
                    }
                  ]
                }
                """));
        AgentStreamAccumulator.Snapshot snapshot = accumulator.accept(parser.parse("""
                {
                  "id": "chatcmpl-1",
                  "object": "chat.completion.chunk",
                  "choices": [
                    {
                      "delta": {
                        "tool_calls": [
                          {
                            "index": 0,
                            "function": {
                              "arguments": ":\\"gollek\\"}"
                            }
                          }
                        ]
                      },
                      "finish_reason": null
                    }
                  ]
                }
                """));

        assertTrue(snapshot.hasToolCalls());
        assertEquals(1, snapshot.toolCalls().size());
        AgentStreamEvent.ToolCall call = snapshot.toolCalls().get(0);
        assertEquals("call_search", call.id());
        assertEquals("knowledge_search", call.name());
        assertEquals("{\"query\":\"gollek\"}", call.arguments());
    }

    @Test
    void replacesResponsesToolArgumentsWithDonePayload() throws Exception {
        AgentStreamAccumulator accumulator = new AgentStreamAccumulator();

        accumulator.accept(parser.parse("""
                {
                  "type": "response.function_call_arguments.delta",
                  "sequence_number": 1,
                  "item_id": "fc_search",
                  "output_index": 1,
                  "delta": "{\\"query\\":\\"gol",
                  "call_id": "call_search"
                }
                """));
        AgentStreamAccumulator.Snapshot snapshot = accumulator.accept(parser.parse("""
                {
                  "type": "response.function_call_arguments.done",
                  "sequence_number": 2,
                  "item_id": "fc_search",
                  "output_index": 1,
                  "arguments": "{\\"query\\":\\"gollek\\"}",
                  "call_id": "call_search"
                }
                """));

        assertEquals(1, snapshot.toolCalls().size());
        AgentStreamEvent.ToolCall call = snapshot.toolCalls().get(0);
        assertEquals("fc_search", call.id());
        assertEquals("call_search", call.callId());
        assertEquals("{\"query\":\"gollek\"}", call.arguments());
    }

    @Test
    void resetClearsAccumulatedState() throws Exception {
        AgentStreamAccumulator accumulator = new AgentStreamAccumulator();
        accumulator.accept(parser.parse("""
                {
                  "type": "error",
                  "sequence_number": 0,
                  "error": {"message": "failed"}
                }
                """));

        assertTrue(accumulator.snapshot().hasErrors());

        accumulator.reset();

        AgentStreamAccumulator.Snapshot snapshot = accumulator.snapshot();
        assertEquals("", snapshot.outputText());
        assertFalse(snapshot.hasErrors());
        assertFalse(snapshot.completed());
    }
}
