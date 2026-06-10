///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//REPOS local,mavencentral
//DEPS tech.kayys.gollek:gollek-sdk-agent:0.1.0-SNAPSHOT

import com.fasterxml.jackson.databind.ObjectMapper;

import tech.kayys.gollek.client.agent.AgentEmbeddingView;
import tech.kayys.gollek.client.agent.AgentIntegrationClient;
import tech.kayys.gollek.client.agent.AgentRequestOptions;
import tech.kayys.gollek.client.agent.AgentStreamAccumulator;
import tech.kayys.gollek.client.agent.AgentStreamEvent;
import tech.kayys.gollek.client.agent.AgentStreamEventParser;

import java.net.http.HttpClient;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Caller Agent bridge example.
 *
 * This script demonstrates the intended product boundary:
 *
 * 1. Gollek is the inference/serving engine.
 * 2. Gollek exposes agent-facing contracts, validation, embeddings, RAG context
 *    injection, MCP tool schemas, and OpenAI-compatible generation APIs.
 * 3. Caller Agent owns planning, vector search, tool execution, approvals,
 *    memory, workflow state, and follow-up steps.
 *
 * Run from gollek/examples/jbang:
 *
 *   jbang run --no-integrations integration/agent_serving_contract_bridge.java --mock
 *
 * For a live Gollek server, publish the SDK first from the gollek project root:
 *
 *   ./gradlew :sdk:gollek-sdk-agent:publishToMavenLocal
 *   cd examples/jbang
 *   jbang run --no-integrations integration/agent_serving_contract_bridge.java --live \
 *       --base-url http://localhost:8080 --api-key community
 */
public class agent_serving_contract_bridge {
    public static void main(String[] args) throws Exception {
        Config config = Config.parse(args);
        if (config.help()) {
            printHelp();
            return;
        }

        GollekAgentGateway gollek = config.live()
                ? HttpGollekAgentGateway.connect(config)
                : new MockGollekAgentGateway();
        AgentRequestOptions trace = AgentRequestOptions.builder()
                .requestId(config.requestId())
                .traceId(config.traceId())
                .sessionId(config.sessionId())
                .userId(config.userId())
                .header("X-Agent-Workflow-Id", config.workflowId())
                .build();

        MockCallerVectorStore callerVectorStore = new MockCallerVectorStore();
        MockCallerToolLoop callerToolLoop = new MockCallerToolLoop();
        MockCallerMemory callerMemory = new MockCallerMemory();

        heading("Caller Agent on Gollek");
        line("mode", config.live() ? "live Gollek server" : "mock serving response");
        line("gollek", config.baseUrl());
        line("workflow", config.workflowId());
        line("trace", config.traceId());
        line("task", config.task());

        heading("1. Discover serving contract");
        Map<String, Object> capabilities = gollek.capabilities(trace);
        Map<String, Object> contract = gollek.contract(trace);
        line("service_role", firstValue(capabilities, contract, "service_role"));
        line("gollek_owns", nested(contract, "boundary", "gollek_owns"));
        line("orchestrator_owns", nested(contract, "boundary", "agent_orchestrator_owns"));

        heading("2. Discover MCP tools");
        Map<String, Object> mcp = gollek.mcpTools(trace);
        List<?> tools = list(mcp.get("tools"));
        line("tool_count", tools.size());
        line("tool_names", toolNames(tools));
        gollek.validateTools(Map.of("tools", tools), trace);
        line("tool_validation", "schema only; execution remains in the caller agent");

        heading("3. Embed task through Gollek");
        Map<String, Object> embeddingRequest = mapOf(
                "model", config.embeddingModel(),
                "input", List.of(config.task()),
                "metadata", Map.of(
                        "orchestrator", "caller-agent",
                        "workflow_id", config.workflowId()));
        gollek.validateRequest("embeddings", embeddingRequest, trace);
        AgentEmbeddingView embedding = gollek.createEmbedding(embeddingRequest, trace);
        List<Double> vector = embedding.firstVector();
        line("embedding_model", embedding.model());
        line("embedding_dimensions", embedding.dimensions());
        line("embedding_trace", embedding.trace());

        heading("4. Search caller-owned RAG store");
        List<RetrievedChunk> chunks = callerVectorStore.search(vector, config.task());
        for (RetrievedChunk chunk : chunks) {
            line("retrieved", chunk.id() + " score=" + chunk.score() + " source=" + chunk.source());
        }

        heading("5. Stream Gollek with caller-selected context");
        Map<String, Object> chatRequest = mapOf(
                "model", config.model(),
                "messages", List.of(
                        Map.of(
                                "role", "system",
                                "content", "Use supplied context. Return tool calls when extra context is needed."),
                        Map.of("role", "user", "content", config.task())),
                "tools", tools,
                "rag_context", chunks.stream().map(RetrievedChunk::toRagItem).toList(),
                "metadata", Map.of(
                        "orchestrator", "caller-agent",
                        "retrieval_owner", "caller-agent",
                        "tool_execution_owner", "caller-agent"));
        gollek.validateRequest("chat", chatRequest, trace);
        List<AgentStreamEvent> streamEvents = new ArrayList<>();
        AgentStreamAccumulator.Snapshot response = gollek.streamChatCompletion(chatRequest, trace, event -> {
            streamEvents.add(event);
            if (event.hasDelta()) {
                line("stream_delta", event.delta());
            }
            if (event.hasToolCalls()) {
                line("stream_tool_preview", toolCallNames(event.toolCalls()));
            }
        });
        line("response_model", config.model());
        line("stream_events", streamEvents.size());
        line("finish_reason", response.finishReason());
        line("assistant", response.outputText());
        line("response_trace", response.trace());

        heading("6. Execute tool declarations in the caller agent");
        callerToolLoop.handle(response.toolCalls());

        heading("7. Record caller-owned memory");
        callerMemory.recordAssistantMessage(response.outputText(), response.trace());
        callerMemory.recordToolCalls(response.toolCalls());
        callerMemory.printSummary();

        heading("Boundary");
        line("gollek", "serving, model runtime, validation, embeddings, MCP discovery, response/tool-call declarations");
        line("caller-agent", "planning, tool execution, RAG store, approvals, memory, workflow state, follow-up loops");
    }

    private interface GollekAgentGateway {
        Map<String, Object> capabilities(AgentRequestOptions trace) throws Exception;

        Map<String, Object> contract(AgentRequestOptions trace) throws Exception;

        Map<String, Object> mcpTools(AgentRequestOptions trace) throws Exception;

        Map<String, Object> validateTools(Map<String, Object> request, AgentRequestOptions trace) throws Exception;

        Map<String, Object> validateRequest(String surface, Map<String, Object> request, AgentRequestOptions trace)
                throws Exception;

        AgentEmbeddingView createEmbedding(Map<String, Object> request, AgentRequestOptions trace) throws Exception;

        AgentStreamAccumulator.Snapshot streamChatCompletion(
                Map<String, Object> request,
                AgentRequestOptions trace,
                Consumer<AgentStreamEvent> eventConsumer) throws Exception;
    }

    private static final class HttpGollekAgentGateway implements GollekAgentGateway {
        private final AgentIntegrationClient agent;

        private HttpGollekAgentGateway(AgentIntegrationClient agent) {
            this.agent = agent;
        }

        static HttpGollekAgentGateway connect(Config config) {
            return new HttpGollekAgentGateway(new AgentIntegrationClient(
                    HttpClient.newHttpClient(),
                    new ObjectMapper(),
                    config.baseUrl(),
                    config.apiKey()));
        }

        @Override
        public Map<String, Object> capabilities(AgentRequestOptions trace) throws Exception {
            return agent.capabilities(trace);
        }

        @Override
        public Map<String, Object> contract(AgentRequestOptions trace) throws Exception {
            return agent.contract(trace);
        }

        @Override
        public Map<String, Object> mcpTools(AgentRequestOptions trace) throws Exception {
            return agent.mcpTools(true, true, trace);
        }

        @Override
        public Map<String, Object> validateTools(Map<String, Object> request, AgentRequestOptions trace)
                throws Exception {
            return agent.validateTools(request, trace);
        }

        @Override
        public Map<String, Object> validateRequest(
                String surface,
                Map<String, Object> request,
                AgentRequestOptions trace) throws Exception {
            return agent.validateRequest(surface, request, trace);
        }

        @Override
        public AgentEmbeddingView createEmbedding(Map<String, Object> request, AgentRequestOptions trace)
                throws Exception {
            return agent.createEmbeddingView(request, trace);
        }

        @Override
        public AgentStreamAccumulator.Snapshot streamChatCompletion(
                Map<String, Object> request,
                AgentRequestOptions trace,
                Consumer<AgentStreamEvent> eventConsumer) throws Exception {
            return agent.streamChatCompletion(request, trace, eventConsumer);
        }
    }

    private static final class MockGollekAgentGateway implements GollekAgentGateway {
        @Override
        public Map<String, Object> capabilities(AgentRequestOptions trace) {
            return mapOf(
                    "service_role", "inference_serving_engine",
                    "surfaces", List.of("chat", "responses", "embeddings", "mcp_tools"),
                    "validation_only", true,
                    "tool_execution", false,
                    "retrieval_execution", false,
                    "trace", traceMap(trace));
        }

        @Override
        public Map<String, Object> contract(AgentRequestOptions trace) {
            return mapOf(
                    "version", "v1",
                    "service_role", "inference_serving_engine",
                    "boundary", Map.of(
                            "gollek_owns", List.of(
                                    "model serving",
                                    "OpenAI-compatible chat and responses",
                                    "embeddings",
                                    "MCP discovery",
                                    "tool-call declarations",
                                    "request validation"),
                            "agent_orchestrator_owns", List.of(
                                    "planning",
                                    "tool execution",
                                    "RAG retrieval",
                                    "memory",
                                    "approval policy",
                                    "workflow state")),
                    "trace", traceMap(trace));
        }

        @Override
        public Map<String, Object> mcpTools(AgentRequestOptions trace) {
            Map<String, Object> querySchema = mapOf(
                    "type", "object",
                    "properties", Map.of(
                            "query", Map.of(
                                    "type", "string",
                                    "description", "Natural-language retrieval query")),
                    "required", List.of("query"));
            Map<String, Object> searchContext = mapOf(
                    "type", "function",
                    "function", mapOf(
                            "name", "search_context",
                            "description", "Search caller-owned knowledge",
                            "parameters", querySchema));
            return mapOf(
                    "object", "gollek.mcp_tools",
                    "compat", "openai",
                    "tools", List.of(searchContext),
                    "trace", traceMap(trace));
        }

        @Override
        public Map<String, Object> validateTools(Map<String, Object> request, AgentRequestOptions trace) {
            return validation("tools", request, trace);
        }

        @Override
        public Map<String, Object> validateRequest(String surface, Map<String, Object> request, AgentRequestOptions trace) {
            return validation(surface, request, trace);
        }

        @Override
        public AgentEmbeddingView createEmbedding(Map<String, Object> request, AgentRequestOptions trace)
                throws Exception {
            return AgentEmbeddingView.fromJson("""
                    {
                      "object": "list",
                      "model": "mock-embedding-model",
                      "data": [
                        {
                          "object": "embedding",
                          "index": 0,
                          "embedding": [0.12, 0.23, 0.34, 0.45, 0.56, 0.67]
                        }
                      ],
                      "metadata": {
                        "gollek_trace": {
                          "request_id": "req-agent-demo",
                          "trace_id": "trace-agent-demo"
                        }
                      },
                      "usage": {
                        "prompt_tokens": 9,
                        "total_tokens": 9
                      }
                    }
                    """);
        }

        @Override
        public AgentStreamAccumulator.Snapshot streamChatCompletion(
                Map<String, Object> request,
                AgentRequestOptions trace,
                Consumer<AgentStreamEvent> eventConsumer) throws Exception {
            AgentStreamEventParser parser = new AgentStreamEventParser(new ObjectMapper());
            AgentStreamAccumulator accumulator = new AgentStreamAccumulator();
            for (String data : List.of(
                    """
                    {
                      "id": "chatcmpl-agent-demo",
                      "object": "chat.completion.chunk",
                      "choices": [
                        {
                          "index": 0,
                          "delta": {"role": "assistant", "content": "Gollek can answer "},
                          "finish_reason": null
                        }
                      ],
                      "metadata": {
                        "gollek_trace": {
                          "request_id": "req-agent-demo",
                          "trace_id": "trace-agent-demo"
                        },
                        "gollek_stream": {"sequence_number": 0}
                      }
                    }
                    """,
                    """
                    {
                      "id": "chatcmpl-agent-demo",
                      "object": "chat.completion.chunk",
                      "choices": [
                        {
                          "index": 0,
                          "delta": {
                            "content": "with supplied context and declares a tool call when the caller agent should fetch more evidence."
                          },
                          "finish_reason": null
                        }
                      ],
                      "metadata": {
                        "gollek_trace": {
                          "request_id": "req-agent-demo",
                          "trace_id": "trace-agent-demo"
                        },
                        "gollek_stream": {"sequence_number": 1}
                      }
                    }
                    """,
                    """
                    {
                      "id": "chatcmpl-agent-demo",
                      "object": "chat.completion.chunk",
                      "choices": [
                        {
                          "index": 0,
                          "delta": {
                            "tool_calls": [
                              {
                                "id": "call_search_1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                  "name": "search_context",
                                  "arguments": "{\\"query\\":\\"agent integration boundary\\"}"
                                }
                              }
                            ]
                          },
                          "finish_reason": null
                        }
                      ],
                      "metadata": {
                        "gollek_trace": {
                          "request_id": "req-agent-demo",
                          "trace_id": "trace-agent-demo"
                        },
                        "gollek_stream": {"sequence_number": 2}
                      }
                    }
                    """,
                    """
                    {
                      "id": "chatcmpl-agent-demo",
                      "object": "chat.completion.chunk",
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                      "metadata": {
                        "gollek_trace": {
                          "request_id": "req-agent-demo",
                          "trace_id": "trace-agent-demo"
                        },
                        "gollek_stream": {
                          "sequence_number": 3,
                          "final": true,
                          "finish_reason": "tool_calls"
                        }
                      },
                      "usage": {
                        "prompt_tokens": 42,
                        "completion_tokens": 18,
                        "total_tokens": 60
                      }
                    }
                    """,
                    AgentStreamEventParser.DONE)) {
                AgentStreamEvent event = parser.parse(data);
                accumulator.accept(event);
                if (eventConsumer != null) {
                    eventConsumer.accept(event);
                }
            }
            return accumulator.snapshot();
        }

        private Map<String, Object> validation(String surface, Map<String, Object> request, AgentRequestOptions trace) {
            return mapOf(
                    "object", "gollek.agent_validation",
                    "surface", surface,
                    "valid", true,
                    "model_invoked", false,
                    "normalized", request == null ? Map.of() : request,
                    "boundary", Map.of(
                            "validation_only", true,
                            "tool_execution", false,
                            "retrieval_execution", false),
                    "trace", traceMap(trace));
        }
    }

    private static final class MockCallerVectorStore {
        List<RetrievedChunk> search(List<Double> vector, String query) {
            double score = vector.isEmpty() ? 0.50 : 0.94;
            return List.of(
                    new RetrievedChunk(
                            "agent-boundary",
                            "Agent Integration Boundary",
                            "docs/agent-integration-boundary.md",
                            score,
                            "Gollek is the serving engine. Caller Agent owns planning, tool execution, memory, "
                                    + "RAG retrieval, approvals, and workflow state."),
                    new RetrievedChunk(
                            "agent-api",
                            "Agent Integration API",
                            "docs/agent-integration-api.md",
                            0.88,
                            "Agent clients can discover capabilities, validate requests, fetch MCP tools, create "
                                    + "embeddings, inject rag_context, and call chat or responses endpoints."));
        }
    }

    private static final class MockCallerToolLoop {
        void handle(List<AgentStreamEvent.ToolCall> calls) throws Exception {
            if (calls.isEmpty()) {
                line("tool_loop", "no tool calls returned");
                return;
            }
            for (AgentStreamEvent.ToolCall call : calls) {
                Map<String, Object> arguments = call.argumentsMap();
                line("tool_call", call.name() + " " + arguments);
                if ("search_context".equals(call.name())) {
                    line("caller_executes", "search_context through caller agent policy, credentials, and audit log");
                    line("tool_result", "2 caller-owned chunks returned; follow-up generation would pass them back to Gollek");
                } else {
                    line("caller_routes", "tool execution delegated to the caller agent registry");
                }
            }
        }
    }

    private static final class MockCallerMemory {
        private final List<Map<String, Object>> events = new ArrayList<>();

        void recordAssistantMessage(String text, Map<String, Object> trace) {
            events.add(mapOf(
                    "type", "assistant_message",
                    "text", text,
                    "trace", trace));
        }

        void recordToolCalls(List<AgentStreamEvent.ToolCall> calls) {
            for (AgentStreamEvent.ToolCall call : calls) {
                events.add(mapOf(
                        "type", "tool_call",
                        "name", call.name(),
                        "arguments", call.arguments()));
            }
        }

        void printSummary() {
            line("memory_owner", "caller-agent");
            line("events_recorded", events.size());
        }
    }

    private record RetrievedChunk(String id, String title, String source, double score, String text) {
        Map<String, Object> toRagItem() {
            return mapOf(
                    "id", id,
                    "title", title,
                    "source", source,
                    "score", score,
                    "text", text);
        }
    }

    private record Config(
            boolean live,
            boolean help,
            String baseUrl,
            String apiKey,
            String model,
            String embeddingModel,
            String task,
            String requestId,
            String traceId,
            String sessionId,
            String userId,
            String workflowId) {

        static Config parse(String[] args) {
            boolean live = false;
            boolean help = false;
            String baseUrl = envOrDefault("GOLLEK_BASE_URL", "http://localhost:8080");
            String apiKey = envOrDefault("GOLLEK_API_KEY", "community");
            String model = envOrDefault("GOLLEK_MODEL", "demo-model");
            String embeddingModel = envOrDefault("GOLLEK_EMBEDDING_MODEL", "demo-embed");
            String task = "Explain the Gollek agent serving boundary.";
            String requestId = "req-agent-demo";
            String traceId = "trace-agent-demo";
            String sessionId = "session-agent-demo";
            String userId = "user-demo";
            String workflowId = "workflow-agent-demo";

            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--help", "-h" -> help = true;
                    case "--mock" -> live = false;
                    case "--live" -> live = true;
                    case "--base-url" -> baseUrl = value(args, ++i, "--base-url");
                    case "--api-key" -> apiKey = value(args, ++i, "--api-key");
                    case "--model" -> model = value(args, ++i, "--model");
                    case "--embedding-model" -> embeddingModel = value(args, ++i, "--embedding-model");
                    case "--task" -> task = value(args, ++i, "--task");
                    case "--request-id" -> requestId = value(args, ++i, "--request-id");
                    case "--trace-id" -> traceId = value(args, ++i, "--trace-id");
                    case "--session-id" -> sessionId = value(args, ++i, "--session-id");
                    case "--user-id" -> userId = value(args, ++i, "--user-id");
                    case "--workflow-id" -> workflowId = value(args, ++i, "--workflow-id");
                    default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
                }
            }
            return new Config(
                    live,
                    help,
                    baseUrl,
                    apiKey,
                    model,
                    embeddingModel,
                    task,
                    requestId,
                    traceId,
                    sessionId,
                    userId,
                    workflowId);
        }
    }

    private static void printHelp() {
        System.out.println("""
                Usage:
                  jbang run --no-integrations integration/agent_serving_contract_bridge.java [--mock|--live] [options]

                Options:
                  --mock                         Run without a live Gollek server (default)
                  --live                         Call a live Gollek serving endpoint
                  --base-url <url>               Gollek base URL (default: http://localhost:8080)
                  --api-key <key>                Gollek API key (default: community)
                  --model <id>                   Generation model id
                  --embedding-model <id>         Embedding model id
                  --task <text>                  User task to send through the bridge
                  --request-id <id>              Request id propagated to Gollek
                  --trace-id <id>                Trace id propagated to Gollek
                  --session-id <id>              Session id propagated to Gollek
                  --user-id <id>                 User id propagated to Gollek
                  --workflow-id <id>             Caller agent workflow id
                """);
    }

    private static String envOrDefault(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? fallback : value;
    }

    private static String value(String[] args, int index, String option) {
        if (index >= args.length || args[index].startsWith("--")) {
            throw new IllegalArgumentException(option + " requires a value");
        }
        return args[index];
    }

    private static Map<String, Object> traceMap(AgentRequestOptions trace) {
        return mapOf(
                "request_id", trace.requestId(),
                "trace_id", trace.traceId(),
                "session_id", trace.sessionId(),
                "user_id", trace.userId());
    }

    private static Map<String, Object> mapOf(Object... pairs) {
        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        if (pairs.length % 2 != 0) {
            throw new IllegalArgumentException("mapOf requires key/value pairs");
        }
        for (int i = 0; i < pairs.length; i += 2) {
            if (!(pairs[i] instanceof String key)) {
                throw new IllegalArgumentException("mapOf keys must be strings");
            }
            out.put(key, pairs[i + 1]);
        }
        return out;
    }

    private static Object firstValue(Map<String, Object> first, Map<String, Object> second, String key) {
        Object value = first.get(key);
        return value == null ? second.get(key) : value;
    }

    @SuppressWarnings("unchecked")
    private static Object nested(Map<String, Object> root, String first, String second) {
        Object value = root.get(first);
        if (value instanceof Map<?, ?> map) {
            return ((Map<String, Object>) map).get(second);
        }
        return null;
    }

    private static List<?> list(Object value) {
        return value instanceof List<?> list ? list : List.of();
    }

    @SuppressWarnings("unchecked")
    private static List<String> toolNames(List<?> tools) {
        List<String> out = new ArrayList<>();
        for (Object tool : tools) {
            if (tool instanceof Map<?, ?> toolMap) {
                Object function = ((Map<String, Object>) toolMap).get("function");
                if (function instanceof Map<?, ?> functionMap) {
                    Object name = ((Map<String, Object>) functionMap).get("name");
                    if (name != null) {
                        out.add(name.toString());
                    }
                }
            }
        }
        return out;
    }

    private static List<String> toolCallNames(List<AgentStreamEvent.ToolCall> calls) {
        List<String> out = new ArrayList<>();
        for (AgentStreamEvent.ToolCall call : calls) {
            if (call.name() != null && !call.name().isBlank()) {
                out.add(call.name());
            }
        }
        return out;
    }

    private static void heading(String text) {
        System.out.println();
        System.out.println("== " + text);
    }

    private static void line(String key, Object value) {
        System.out.println("  " + key + ": " + value);
    }
}
