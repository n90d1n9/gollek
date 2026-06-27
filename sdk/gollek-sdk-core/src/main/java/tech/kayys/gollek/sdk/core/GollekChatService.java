package tech.kayys.gollek.sdk.core;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.tool.ToolDefinition;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Shared chat service facade that abstracts away the construction of
 * InferenceRequests, allowing a unified multi-turn chat and tool-use
 * experience across CLI, REST API, gRPC, and embedded environments (Wayang).
 */
public final class GollekChatService {

    public Multi<StreamingInferenceChunk> streamChat(
            GollekSdk sdk, 
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools, 
            ChatParams p) {

        InferenceRequest req = buildRequest(modelId, systemPrompt, history, tools, p, true);
        return sdk.streamCompletion(req);
    }

    public Uni<InferenceResponse> chat(
            GollekSdk sdk, 
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools, 
            ChatParams p) {

        InferenceRequest req = buildRequest(modelId, systemPrompt, history, tools, p, false);
        // InferenceEngine's createCompletion is synchronous, createCompletionAsync is CompletableFuture,
        // so we wrap the CompletableFuture in a Uni.
        return Uni.createFrom().completionStage(() -> sdk.createCompletionAsync(req));
    }

    private InferenceRequest buildRequest(
            String modelId, 
            String systemPrompt,
            List<Message> history, 
            List<ToolDefinition> tools,
            ChatParams p, 
            boolean streaming) {

        List<Message> messages = new ArrayList<>();
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            messages.add(Message.system(systemPrompt));
        }
        if (history != null) {
            messages.addAll(history);
        }

        InferenceRequest.Builder reqBuilder = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(modelId)
                .messages(messages)
                .streaming(streaming)
                .parameter("temperature", p.temperature())
                .parameter("max_tokens", p.maxTokens())
                .parameter("top_p", p.topP())
                .parameter("repeat_penalty", p.repeatPenalty());

        if (tools != null && !tools.isEmpty()) {
            reqBuilder.tools(tools);
        }

        return reqBuilder.build();
    }
}
