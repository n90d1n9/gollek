package tech.kayys.gollek.client.agent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Incrementally assembles OpenAI-compatible agent stream events.
 *
 * <p>The accumulator keeps client-side stream state only. It does not execute
 * tools, authorize calls, plan next steps, or own memory/workflow state.
 */
public final class AgentStreamAccumulator {
    private final StringBuilder outputText = new StringBuilder();
    private final Map<String, MutableToolCall> toolCalls = new LinkedHashMap<>();
    private final Map<Integer, String> toolCallKeysByIndex = new LinkedHashMap<>();
    private final List<String> errors = new ArrayList<>();
    private final Map<String, Object> trace = new LinkedHashMap<>();
    private final Map<String, Object> metadata = new LinkedHashMap<>();

    private AgentStreamEvent.Usage usage;
    private String finishReason;
    private boolean completed;

    /**
     * Adds an event to the accumulated stream state and returns the current snapshot.
     */
    public Snapshot accept(AgentStreamEvent event) {
        if (event == null) {
            return snapshot();
        }

        accumulateText(event);
        accumulateToolCalls(event);
        accumulateLifecycle(event);
        return snapshot();
    }

    /**
     * Returns the current immutable view of accumulated stream state.
     */
    public Snapshot snapshot() {
        List<AgentStreamEvent.ToolCall> calls = toolCalls.values().stream()
                .map(MutableToolCall::snapshot)
                .toList();
        return new Snapshot(
                outputText.toString(),
                calls,
                usage,
                finishReason,
                completed,
                List.copyOf(errors),
                Map.copyOf(trace),
                Map.copyOf(metadata));
    }

    /**
     * Clears all accumulated state so the instance can be reused.
     */
    public void reset() {
        outputText.setLength(0);
        toolCalls.clear();
        toolCallKeysByIndex.clear();
        errors.clear();
        trace.clear();
        metadata.clear();
        usage = null;
        finishReason = null;
        completed = false;
    }

    private void accumulateText(AgentStreamEvent event) {
        if (event.hasDelta() && isTextDelta(event)) {
            outputText.append(event.delta());
        }
        if (notBlank(event.text()) && event.isOutputDone()) {
            replaceOutputText(event.text());
        }
        if (notBlank(event.outputText())) {
            replaceOutputText(event.outputText());
        }
    }

    private boolean isTextDelta(AgentStreamEvent event) {
        return !event.hasToolCalls()
                && (event.surface() == AgentStreamEvent.Surface.CHAT_COMPLETIONS
                || "response.output_text.delta".equals(event.type())
                || event.type() == null);
    }

    private void replaceOutputText(String text) {
        outputText.setLength(0);
        outputText.append(text);
    }

    private void accumulateToolCalls(AgentStreamEvent event) {
        for (AgentStreamEvent.ToolCall call : event.toolCalls()) {
            String key = toolCallKey(call);
            toolCalls.computeIfAbsent(key, ignored -> new MutableToolCall()).merge(call);
            if (call.index() >= 0) {
                toolCallKeysByIndex.put(call.index(), key);
            }
        }
    }

    private String toolCallKey(AgentStreamEvent.ToolCall call) {
        if (call.index() >= 0 && toolCallKeysByIndex.containsKey(call.index())) {
            return toolCallKeysByIndex.get(call.index());
        }
        if (notBlank(call.id())) {
            return "id:" + call.id();
        }
        if (notBlank(call.callId())) {
            return "call:" + call.callId();
        }
        return "index:" + call.index();
    }

    private void accumulateLifecycle(AgentStreamEvent event) {
        if (event.usage() != null) {
            usage = event.usage();
        }
        if (notBlank(event.finishReason())) {
            finishReason = event.finishReason();
        }
        if (event.isCompleted()) {
            completed = true;
        }
        if (event.isError()) {
            errors.add(event.errorMessage());
        }
        trace.putAll(event.trace());
        metadata.putAll(event.metadata());
    }

    private static boolean notBlank(String value) {
        return value != null && !value.isBlank();
    }

    /**
     * Immutable accumulated stream state.
     */
    public record Snapshot(
            String outputText,
            List<AgentStreamEvent.ToolCall> toolCalls,
            AgentStreamEvent.Usage usage,
            String finishReason,
            boolean completed,
            List<String> errors,
            Map<String, Object> trace,
            Map<String, Object> metadata) {

        public boolean hasToolCalls() {
            return toolCalls != null && !toolCalls.isEmpty();
        }

        public boolean hasErrors() {
            return errors != null && !errors.isEmpty();
        }
    }

    private static final class MutableToolCall {
        private String id;
        private int index = -1;
        private String type;
        private String name;
        private final StringBuilder arguments = new StringBuilder();
        private String callId;
        private String status;
        private com.fasterxml.jackson.databind.JsonNode raw;

        private void merge(AgentStreamEvent.ToolCall call) {
            id = firstNonBlank(id, call.id());
            index = index >= 0 ? index : call.index();
            type = firstNonBlank(call.type(), type);
            name = firstNonBlank(name, call.name());
            if (call.hasArguments()) {
                if (isFinalToolCall(call)) {
                    arguments.setLength(0);
                }
                arguments.append(call.arguments());
            }
            callId = firstNonBlank(callId, call.callId());
            status = firstNonBlank(call.status(), status);
            raw = call.raw() != null ? call.raw() : raw;
        }

        private AgentStreamEvent.ToolCall snapshot() {
            return new AgentStreamEvent.ToolCall(
                    id,
                    index,
                    type,
                    name,
                    arguments.isEmpty() ? null : arguments.toString(),
                    callId,
                    status,
                    raw);
        }
    }

    private static String firstNonBlank(String first, String second) {
        return notBlank(first) ? first : second;
    }

    private static boolean isFinalToolCall(AgentStreamEvent.ToolCall call) {
        return call.type() != null && call.type().endsWith(".done")
                || "completed".equals(call.status());
    }
}
