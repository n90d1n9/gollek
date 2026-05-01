package tech.kayys.gollek.sdk.session;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class ChatSessionImpl implements ChatSession {

    private final String sessionId;
    private final GollekSdk sdk;
    private final List<Message> history = new ArrayList<>();
    
    private String modelId;
    private String providerId;
    private Map<String, Object> defaultParameters = new HashMap<>();
    private boolean autoContinue = true;

    // Stats
    private final Instant startTime = Instant.now();
    private final AtomicInteger totalRequests = new AtomicInteger(0);
    private final AtomicInteger totalTokens = new AtomicInteger(0);
    private final AtomicLong totalDurationMs = new AtomicLong(0);
    private final AtomicInteger totalErrors = new AtomicInteger(0);
    private final Map<String, int[]> perModelStats = new HashMap<>();
    private final Map<String, int[]> perProviderStats = new HashMap<>();

    public ChatSessionImpl(GollekSdk sdk, String modelId, String providerId) {
        this.sdk = sdk;
        this.modelId = modelId;
        this.providerId = providerId;
        this.sessionId = UUID.randomUUID().toString();
    }

    @Override
    public String getSessionId() {
        return sessionId;
    }

    @Override
    public String getModelId() {
        return modelId;
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    @Override
    public void setDefaultParameters(Map<String, Object> parameters) {
        if (parameters != null) {
            this.defaultParameters = new HashMap<>(parameters);
        }
    }

    @Override
    public void setSystemPrompt(String systemPrompt) {
        if (systemPrompt != null && !systemPrompt.isBlank()) {
            synchronized (history) {
                // If there's already a system prompt at the beginning, replace it
                if (!history.isEmpty() && history.get(0).getRole() == Message.Role.SYSTEM) {
                    history.set(0, Message.system(systemPrompt));
                } else {
                    history.add(0, Message.system(systemPrompt));
                }
            }
        }
    }

    @Override
    public void setAutoContinue(boolean autoContinue) {
        this.autoContinue = autoContinue;
    }

    @Override
    public void addMessage(Message message) {
        synchronized (history) {
            history.add(message);
        }
    }

    @Override
    public List<Message> getHistory() {
        synchronized (history) {
            return new ArrayList<>(history);
        }
    }

    @Override
    public void reset() {
        synchronized (history) {
            history.clear();
        }
    }

    @Override
    public InferenceResponse send(String prompt) throws SdkException {
        return send(InferenceRequest.builder()
                .model(modelId)
                .messages(getHistoryWithPrompt(prompt))
                .preferredProvider(providerId)
                .build());
    }

    @Override
    public InferenceResponse send(InferenceRequest request) throws SdkException {
        long start = System.currentTimeMillis();
        try {
            InferenceRequest enriched = enrichRequest(request);
            InferenceResponse response = sdk.createCompletion(enriched);
            
            long duration = System.currentTimeMillis() - start;
            recordStats(response.getTokensUsed(), duration, false);
            
            String content = response.getContent();
            StringBuilder fullResponse = new StringBuilder(content);

            // Handle auto-continue for synchronous requests
            if (autoContinue && looksTruncated(content)) {
                String continuation = requestContinuation();
                if (continuation != null && !continuation.isBlank()) {
                    fullResponse.append(continuation);
                    content = fullResponse.toString();
                }
            }

            addMessage(Message.user(getLastUserPrompt(enriched)));
            addMessage(Message.assistant(content));
            
            return response.toBuilder().content(content).build();
        } catch (Exception e) {
            totalErrors.incrementAndGet();
            throw e;
        }
    }

    private String requestContinuation() throws SdkException {
        List<Message> continuationMessages = getHistory();
        continuationMessages.add(Message.user("Continue from exactly where you stopped."));

        InferenceRequest request = InferenceRequest.builder()
                .requestId(UUID.randomUUID().toString())
                .model(modelId)
                .messages(continuationMessages)
                .preferredProvider(providerId)
                .cacheBypass(true)
                .build();

        return sdk.createCompletion(enrichRequest(request)).getContent();
    }

    private boolean looksTruncated(String content) {
        if (content == null || content.length() < 24)
            return false;
        if (hasEndOfSequenceToken(content)) {
            return false;
        }
        String trimmed = stripSpecialTokens(content).trim();
        if (trimmed.isEmpty()) {
            return false;
        }
        return !(trimmed.endsWith(".") || trimmed.endsWith("!")
                || trimmed.endsWith("?") || trimmed.endsWith("```")
                || trimmed.endsWith("\n"));
    }

    private boolean hasEndOfSequenceToken(String content) {
        if (content == null) return false;
        return SPECIAL_TOKEN_PATTERN.matcher(content).find();
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(String prompt) throws SdkException {
        return stream(InferenceRequest.builder()
                .model(modelId)
                .messages(getHistoryWithPrompt(prompt))
                .preferredProvider(providerId)
                .streaming(true)
                .build());
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) throws SdkException {
        InferenceRequest enriched = enrichRequest(request);
        long start = System.currentTimeMillis();
        
        StringBuilder fullContent = new StringBuilder();
        AtomicInteger tokens = new AtomicInteger(0);

        return sdk.streamCompletion(enriched)
                .onItem().transform(chunk -> {
                    if (chunk.getDelta() != null) {
                        String cleanDelta = stripSpecialTokens(chunk.getDelta());
                        fullContent.append(chunk.getDelta());
                        tokens.incrementAndGet();
                        return new StreamingInferenceChunk(
                                chunk.requestId(),
                                chunk.index(),
                                chunk.modality(),
                                cleanDelta,
                                chunk.imageDeltaBase64(),
                                chunk.finished(),
                                chunk.finishReason(),
                                chunk.usage(),
                                chunk.emittedAt(),
                                chunk.metadata()
                        );
                    }
                    return chunk;
                })
                .onFailure().invoke(err -> {
                    totalErrors.incrementAndGet();
                })
                .onCompletion().invoke(() -> {
                    long duration = System.currentTimeMillis() - start;
                    recordStats(tokens.get(), duration, false);
                    
                    String content = fullContent.toString();
                    // Note: Auto-continue for streaming is more complex to implement here 
                    // as we return Multi. For now, we only handle it in synchronous send().
                    
                    addMessage(Message.user(getLastUserPrompt(enriched)));
                    addMessage(Message.assistant(content));
                });
    }

    private static final java.util.regex.Pattern SPECIAL_TOKEN_PATTERN =
            java.util.regex.Pattern.compile(
                    "<\\|im_start\\|>(?:system|user|assistant)?|<\\|im_end\\|>|<\\|endoftext\\|>|</s>|<s>|\\s?\\[INST\\]|\\s?\\[/INST\\]|<\\|eot_id\\|>|<\\|start_header_id\\|>.*?<\\|end_header_id\\|>");

    private String stripSpecialTokens(String delta) {
        if (delta == null || delta.isEmpty()) {
            return "";
        }
        return SPECIAL_TOKEN_PATTERN.matcher(delta).replaceAll("");
    }

    @Override
    public SessionStats getStats() {
        return new SessionStats(
                startTime,
                Duration.between(startTime, Instant.now()).toSeconds(),
                totalRequests.get(),
                totalTokens.get(),
                totalDurationMs.get(),
                totalErrors.get(),
                new HashMap<>(perModelStats),
                new HashMap<>(perProviderStats)
        );
    }

    @Override
    public void close() {
        // Cleanup if needed
    }

    private List<Message> getHistoryWithPrompt(String prompt) {
        List<Message> currentHistory = getHistory();
        currentHistory.add(Message.user(prompt));
        return currentHistory;
    }

    private InferenceRequest enrichRequest(InferenceRequest request) {
        InferenceRequest.Builder builder = request.toBuilder();
        if (request.getModel() == null) builder.model(modelId);
        if (request.getPreferredProvider().isEmpty()) builder.preferredProvider(providerId);
        
        // Apply default parameters
        for (Map.Entry<String, Object> entry : defaultParameters.entrySet()) {
            if (!request.getParameters().containsKey(entry.getKey())) {
                builder.parameter(entry.getKey(), entry.getValue());
            }
        }

        // Add session ID to parameters
        Map<String, Object> params = new HashMap<>(request.getParameters());
        params.put("session_id", sessionId);
        builder.parameters(params);
        
        return builder.build();
    }

    private String getLastUserPrompt(InferenceRequest request) {
        return request.getMessages().stream()
                .filter(m -> m.getRole() == Message.Role.USER)
                .reduce((first, second) -> second)
                .map(Message::getContent)
                .orElse("");
    }

    private void recordStats(int tokens, long durationMs, boolean error) {
        totalRequests.incrementAndGet();
        totalTokens.addAndGet(tokens);
        totalDurationMs.addAndGet(durationMs);
        
        String mId = modelId != null ? modelId : "unknown";
        perModelStats.computeIfAbsent(mId, k -> new int[3]);
        perModelStats.get(mId)[0]++;
        perModelStats.get(mId)[1] += tokens;
        if (error) perModelStats.get(mId)[2]++;

        String pId = providerId != null ? providerId : "auto";
        perProviderStats.computeIfAbsent(pId, k -> new int[3]);
        perProviderStats.get(pId)[0]++;
        perProviderStats.get(pId)[1] += tokens;
        if (error) perProviderStats.get(pId)[2]++;
    }
}
