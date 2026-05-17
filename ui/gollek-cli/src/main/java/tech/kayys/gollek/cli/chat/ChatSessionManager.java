package tech.kayys.gollek.cli.chat;

import io.quarkus.arc.Arc;
import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.sdk.session.ChatSession;
import tech.kayys.gollek.sdk.session.ChatSessionImpl;
import tech.kayys.gollek.sdk.session.ChatSessionFactory;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderRequests;
import tech.kayys.gollek.spi.provider.StreamingProvider;

import java.io.PrintWriter;
import java.time.Duration;
import java.util.*;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CLI-specific wrapper around SDK ChatSession.
 * Manages UI rendering and CLI hooks.
 */
@Dependent
public class ChatSessionManager {

    private ChatSession sdkSession;
    private final ChatSessionFactory sessionFactory;
    @Inject
    ProviderRegistry providerRegistry;
    
    private String modelId;
    private String providerId;
    private String modelPathOverride;
    private boolean enableSession;

    // UI/Output hooks
    private ChatUIRenderer uiRenderer;
    private PrintWriter fileWriter;
    private boolean quiet;
    private boolean autoContinue = true;
    private int maxTokens = 256;
    private double temperature = 0.2;
    private volatile String lastExecutionRoute = "none";
    private volatile String lastProviderDescriptor = "unknown";
    private volatile String lastExecutionError = null;
    private volatile Map<String, Object> lastExecutionMetadata = Map.of();
    private volatile InferenceRequest lastPreparedRequest;
    private volatile boolean lastPreparedRequestStreaming;
    private volatile boolean lastPreparedRequestJsonSse;

    private final GollekSdk sdk;

    @Inject
    public ChatSessionManager(GollekSdk sdk, ChatSessionFactory sessionFactory) {
        this.sdk = sdk;
        this.sessionFactory = sessionFactory;
    }

    public void initialize(String modelId, String providerId, String modelPathOverride, boolean enableSession, boolean forceGguf) {
        this.modelId = modelId;
        this.providerId = providerId;
        this.modelPathOverride = modelPathOverride;
        this.enableSession = enableSession;
        
        this.sdkSession = createSession(modelId, providerId);
    }

    public void reset() {
        if (sdkSession != null) {
            sdkSession.reset();
        }
    }

    public void switchProvider(String providerId) throws SdkException {
        this.providerId = providerId;
        sdk.setPreferredProvider(providerId);
    }

    public void switchModel(String newModelId) {
        this.modelId = newModelId;
        this.modelPathOverride = null;
        this.sdkSession = createSession(newModelId, providerId);
    }

    public String getModelId() {
        return modelId;
    }

    public String getProviderId() {
        return providerId;
    }

    public String getSessionId() {
        return sdkSession != null ? sdkSession.getSessionId() : null;
    }

    public boolean isSessionEnabled() {
        return sessionEnabledForExecution();
    }

    private ChatSession createSession(String modelId, String providerId) {
        return new ChatSessionImpl(sdk, modelId, providerId, enableSession);
    }

    public void setInferenceParams(boolean autoContinue, int maxTokens, double temperature) {
        this.autoContinue = autoContinue;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        
        if (sdkSession != null) {
            sdkSession.setAutoContinue(autoContinue);
            Map<String, Object> params = new HashMap<>();
            params.put("max_tokens", maxTokens);
            params.put("temperature", temperature);
            sdkSession.setDefaultParameters(params);
        }
    }

    public void setSystemPrompt(String systemPrompt) {
        if (sdkSession != null) {
            sdkSession.setSystemPrompt(systemPrompt);
        }
    }

    public void setUIHooks(ChatUIRenderer uiRenderer, PrintWriter fileWriter, boolean quiet) {
        this.uiRenderer = uiRenderer;
        this.fileWriter = fileWriter;
        this.quiet = quiet;
    }

    public void addMessage(Message message) {
        if (sdkSession != null) {
            sdkSession.addMessage(message);
        }
    }

    public List<Message> getHistory() {
        return sdkSession != null ? sdkSession.getHistory() : List.of();
    }

    public void clearHistory() {
        if (sdkSession != null) {
            sdkSession.reset();
        }
    }

    public void executeInference(InferenceRequest.Builder reqBuilder, boolean stream, boolean enableJsonSse) {
        if (sdkSession == null) {
            uiRenderer.printError("Session not initialized", quiet);
            return;
        }

        reqBuilder.model(modelId)
                .preferredProvider(providerId)
                .maxTokens(maxTokens)
                .temperature(temperature);

        if (modelPathOverride != null && !modelPathOverride.isBlank()) {
            reqBuilder.parameter("model_path", modelPathOverride);
        }

        InferenceRequest request = reqBuilder.build();
        InferenceRequest preparedRequest = ensureSessionBinding(activeSession().prepareRequest(request));
        rememberPreparedRequest(preparedRequest, stream, enableJsonSse);

        try {
            executePreparedRequest(preparedRequest, stream, enableJsonSse, false);
        } catch (SdkException e) {
            uiRenderer.printError("Inference failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Inference interrupted", quiet);
        }
    }

    public void retryLastRequest() {
        if (lastPreparedRequest == null) {
            uiRenderer.printError("No previous request is available to retry", quiet);
            return;
        }
        if (!shouldUseDirectProviderPath(lastPreparedRequest)) {
            uiRenderer.printError("Retry is currently only supported on the local direct provider path", quiet);
            return;
        }

        InferenceRequest retryRequest = ensureSessionBinding(lastPreparedRequest.toBuilder()
                .requestId(java.util.UUID.randomUUID().toString())
                .build());
        try {
            executePreparedRequest(retryRequest, lastPreparedRequestStreaming, lastPreparedRequestJsonSse, true);
        } catch (SdkException e) {
            uiRenderer.printError("Retry failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Retry interrupted", quiet);
        }
    }

    private void executePreparedRequest(
            InferenceRequest preparedRequest,
            boolean stream,
            boolean enableJsonSse,
            boolean replaceLastAssistantResponse) throws InterruptedException, SdkException {
        boolean directProviderPath = shouldUseDirectProviderPath(preparedRequest);
        String plannedRoute = directProviderPath
                ? (stream ? "provider-direct-stream" : "provider-direct-sync")
                : (stream ? "sdk-session-stream" : "sdk-session-sync");
        recordExecutionSnapshot(
                plannedRoute,
                directProviderPath ? describeDirectProvider() : (providerId != null ? providerId : "sdk"),
                Map.of(
                        "execution_stage", "planned",
                        "retry", replaceLastAssistantResponse));

        if (directProviderPath) {
            if (stream) {
                executeStreamingDirect(preparedRequest, enableJsonSse, replaceLastAssistantResponse);
            } else {
                executeNonStreamingDirect(preparedRequest, replaceLastAssistantResponse);
            }
            return;
        }
        if (stream) {
            executeStreaming(preparedRequest, enableJsonSse);
        } else {
            executeNonStreaming(preparedRequest);
        }
    }

    private void rememberPreparedRequest(InferenceRequest preparedRequest, boolean stream, boolean enableJsonSse) {
        this.lastPreparedRequest = preparedRequest;
        this.lastPreparedRequestStreaming = stream;
        this.lastPreparedRequestJsonSse = enableJsonSse;
    }

    private void executeStreaming(InferenceRequest request, boolean enableJsonSse) throws InterruptedException, SdkException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        AtomicLong firstTokenTime = new AtomicLong(0);
        AtomicLong lastTokenTime = new AtomicLong(0);
        AtomicLong sumItl = new AtomicLong(0);
        java.util.concurrent.atomic.AtomicReference<Map<String, Object>> metadataRef = new java.util.concurrent.atomic.AtomicReference<>(Map.of());
        StringBuilder fullResponse = new StringBuilder();
        boolean[] quantCachePrinted = { false };

        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet && !enableJsonSse && !jsonMode) {
            spinner.start();
        }

        sdkSession.stream(request)
                .subscribe().with(
                        chunk -> {
                            if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                metadataRef.set(Map.copyOf(chunk.metadata()));
                            }
                            if (!quantCachePrinted[0] && chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                printQuantCacheInfo(chunk.metadata(), enableJsonSse || jsonMode);
                                quantCachePrinted[0] = true;
                            }
                            String delta = chunk.getDelta();
                            if (delta == null) return;
                            
                            fullResponse.append(delta);
                            tokenCount.incrementAndGet();
                            
                            if (delta.isEmpty()) return;

                            long now = System.currentTimeMillis();
                            if (firstTokenTime.compareAndSet(0, now)) {
                                spinner.stop();
                                if (!enableJsonSse && !jsonMode) {
                                    uiRenderer.printAssistantPrefix(quiet, true);
                                }
                            } else {
                                sumItl.addAndGet(now - lastTokenTime.get());
                            }
                            lastTokenTime.set(now);
                            
                            if (fileWriter != null) {
                                fileWriter.print(delta);
                                fileWriter.flush();
                            } else if (enableJsonSse) {
                                printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                            } else if (!jsonMode) {
                                System.out.print(delta);
                                System.out.flush();
                            }
                        },
                        error -> {
                            spinner.stop();
                            recordExecutionFailure("sdk-session-stream", providerId != null ? providerId : "sdk", summarizeThrowable(error));
                            uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                            latch.countDown();
                        },
                        () -> {
                            spinner.stop();
                            long duration = System.currentTimeMillis() - startTime;
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            recordExecutionSnapshot("sdk-session-stream", providerId != null ? providerId : "sdk", metadataRef.get());
                            
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else if (jsonMode) {
                                System.out.println();
                                printJsonModeResponse(request, fullResponse.toString(), tokenCount.get(), duration / 1000.0, tps);
                            } else {
                                System.out.println();
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps,
                                        ttftMillis(metadataRef.get(), startTime, firstTokenTime), quiet);
                            }

                            latch.countDown();
                        });

        latch.await();
    }

    private void executeStreamingDirect(
            InferenceRequest request,
            boolean enableJsonSse,
            boolean replaceLastAssistantResponse) throws InterruptedException, SdkException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        AtomicLong firstTokenTime = new AtomicLong(0);
        AtomicLong lastTokenTime = new AtomicLong(0);
        AtomicLong sumItl = new AtomicLong(0);
        java.util.concurrent.atomic.AtomicReference<Map<String, Object>> metadataRef = new java.util.concurrent.atomic.AtomicReference<>(Map.of());
        StringBuilder fullResponse = new StringBuilder();
        boolean[] quantCachePrinted = { false };
        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet && !enableJsonSse && !jsonMode) {
            spinner.start();
        }

        try {
            StreamingProvider provider = requireStreamingProvider();
            provider.inferStream(buildDirectProviderRequest(request, true))
                    .subscribe().with(
                            chunk -> {
                                if (chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                    metadataRef.set(Map.copyOf(chunk.metadata()));
                                }
                                if (!quantCachePrinted[0] && chunk.metadata() != null && !chunk.metadata().isEmpty()) {
                                    printQuantCacheInfo(chunk.metadata(), enableJsonSse || jsonMode);
                                    quantCachePrinted[0] = true;
                                }
                                String delta = chunk.getDelta();
                                if (delta == null) {
                                    return;
                                }

                                fullResponse.append(delta);
                                tokenCount.incrementAndGet();

                                if (delta.isEmpty()) {
                                    return;
                                }

                                long now = System.currentTimeMillis();
                                if (firstTokenTime.compareAndSet(0, now)) {
                                    spinner.stop();
                                    if (!enableJsonSse && !jsonMode) {
                                        uiRenderer.printAssistantPrefix(quiet, true);
                                    }
                                } else {
                                    sumItl.addAndGet(now - lastTokenTime.get());
                                }
                                lastTokenTime.set(now);

                                if (fileWriter != null) {
                                    fileWriter.print(delta);
                                    fileWriter.flush();
                                } else if (enableJsonSse) {
                                    printOpenAiSseDelta(request.getRequestId(), request.getModel(), delta);
                                } else if (!jsonMode) {
                                    System.out.print(delta);
                                    System.out.flush();
                                }
                            },
                            error -> {
                                spinner.stop();
                                activeSession().recordExternalError(System.currentTimeMillis() - startTime);
                                recordExecutionFailure("provider-direct-stream", describeDirectProvider(), summarizeThrowable(error));
                                uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                                latch.countDown();
                            },
                            () -> {
                                spinner.stop();
                                long duration = System.currentTimeMillis() - startTime;
                                double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                                activeSession().recordExternalAssistantResponse(
                                        fullResponse.toString(),
                                        tokenCount.get(),
                                        duration,
                                        replaceLastAssistantResponse);
                                recordExecutionSnapshot("provider-direct-stream", describeDirectProvider(), metadataRef.get());

                                if (enableJsonSse) {
                                    printOpenAiSseFinal(request.getRequestId(), request.getModel());
                                } else if (jsonMode) {
                                    System.out.println();
                                    printJsonModeResponse(request, fullResponse.toString(), tokenCount.get(), duration / 1000.0, tps);
                                } else {
                                    System.out.println();
                                    uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps,
                                            ttftMillis(metadataRef.get(), startTime, firstTokenTime), quiet);
                                }
                                latch.countDown();
                            });
        } catch (RuntimeException e) {
            spinner.stop();
            activeSession().recordExternalError(System.currentTimeMillis() - startTime);
            recordExecutionFailure("provider-direct-stream", describeDirectProvider(), summarizeThrowable(e));
            throw directProviderFailure("streaming", e);
        }

        latch.await();
    }

    private void executeNonStreaming(InferenceRequest request) throws SdkException {
        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet) spinner.start();
        long startTime = System.currentTimeMillis();
        
        try {
            InferenceResponse response = sdkSession.send(request);
            long duration = System.currentTimeMillis() - startTime;
            double tps = response.getTokensUsed() / (Math.max(1, duration) / 1000.0);
            recordExecutionSnapshot("sdk-session-sync", providerId != null ? providerId : "sdk", response.getMetadata());
            
            boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

            if (jsonMode) {
                printJsonModeResponse(request, response.getContent(), response.getTokensUsed(), duration / 1000.0, tps);
            } else {
                printQuantCacheInfo(response.getMetadata(), false);
                uiRenderer.printAssistantPrefix(quiet, false);
                System.out.println(response.getContent());
                uiRenderer.printStats(response.getTokensUsed(), duration / 1000.0, tps,
                        ttftMillis(response.getMetadata()), quiet);
            }
        } catch (SdkException e) {
            recordExecutionFailure("sdk-session-sync", providerId != null ? providerId : "sdk", summarizeThrowable(e));
            throw e;
        } finally {
            spinner.stop();
        }
    }

    private void executeNonStreamingDirect(InferenceRequest request, boolean replaceLastAssistantResponse) throws SdkException {
        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet) spinner.start();
        long startTime = System.currentTimeMillis();

        try {
            String providerDescriptor = describeDirectProvider();
            InferenceResponse response = requireProvider()
                    .infer(buildDirectProviderRequest(request, false))
                    .await()
                    .atMost(Duration.ofSeconds(300));
            long duration = System.currentTimeMillis() - startTime;
            int tokens = response.getTokensUsed() > 0 ? response.getTokensUsed()
                    : (response.getOutputTokens() > 0 ? response.getOutputTokens() : 0);
            double tps = tokens / (Math.max(1, duration) / 1000.0);
            activeSession().recordExternalAssistantResponse(
                    response.getContent(),
                    tokens,
                    duration,
                    replaceLastAssistantResponse);
            recordExecutionSnapshot("provider-direct-sync", providerDescriptor, response.getMetadata());

            boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;
            if (jsonMode) {
                printJsonModeResponse(request, response.getContent(), tokens, duration / 1000.0, tps);
            } else {
                printQuantCacheInfo(response.getMetadata(), false);
                uiRenderer.printAssistantPrefix(quiet, false);
                System.out.println(response.getContent());
                uiRenderer.printStats(tokens, duration / 1000.0, tps,
                        ttftMillis(response.getMetadata()), quiet);
            }
        } catch (RuntimeException e) {
            activeSession().recordExternalError(System.currentTimeMillis() - startTime);
            recordExecutionFailure("provider-direct-sync", describeDirectProvider(), summarizeThrowable(e));
            throw directProviderFailure("sync", e);
        } finally {
            spinner.stop();
        }
    }

    private static Double ttftMillis(Map<String, Object> metadata) {
        return metadataDouble(metadata, "bench.ttft_ms");
    }

    private static Double ttftMillis(Map<String, Object> metadata, long startTimeMs, AtomicLong firstTokenTimeMs) {
        Double metadataTtft = ttftMillis(metadata);
        if (metadataTtft != null) {
            return metadataTtft;
        }
        long first = firstTokenTimeMs != null ? firstTokenTimeMs.get() : 0L;
        if (first <= 0L || first < startTimeMs) {
            return null;
        }
        return (double) (first - startTimeMs);
    }

    private static Double metadataDouble(Map<String, Object> metadata, String key) {
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }
        Object value = metadata.get(key);
        return value instanceof Number number ? number.doubleValue() : null;
    }

    private void printQuantCacheInfo(Map<String, Object> metadata, boolean suppressForStructuredOutput) {
        if (suppressForStructuredOutput || metadata == null || metadata.isEmpty()) {
            return;
        }
        Object state = metadata.get("quant_cache_state");
        if (state == null) {
            return;
        }
        StringBuilder line = new StringBuilder("Quant cache: ").append(state);
        Object path = metadata.get("quant_cache_path");
        if (path != null) {
            line.append(" (").append(path).append(")");
        }
        System.out.println(line);
        System.out.println("--------------------------------------------------");
    }

    private void printOpenAiSseDelta(String requestId, String model, String delta) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format(
                "{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
                requestId, created, model != null ? model : "", escapeJson(delta));
        System.out.println("data: " + payload);
    }

    private void printOpenAiSseFinal(String requestId, String model) {
        long created = System.currentTimeMillis() / 1000L;
        String payload = String.format(
                "{\"id\":\"chatcmpl-%s\",\"object\":\"chat.completion.chunk\",\"created\":%d,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}",
                requestId, created, model != null ? model : "");
        System.out.println("data: " + payload);
        System.out.println("data: [DONE]");
    }

    private void printJsonModeResponse(InferenceRequest request, String content, int tokens, double duration, double tps) {
        String lastUserPrompt = "";
        List<Message> history = getHistory();
        if (!history.isEmpty()) {
            for (int i = history.size() - 1; i >= 0; i--) {
                if (history.get(i).getRole() == Message.Role.USER) {
                    lastUserPrompt = history.get(i).getContent();
                    break;
                }
            }
        }

        String json = String.format(
                "{\"prompt\":\"%s\",\"model\":\"%s\",\"response\":\"%s\",\"stats\":{\"tokens\":%d,\"duration_s\":%.2f,\"speed_tps\":%.2f}}",
                escapeJson(lastUserPrompt),
                request.getModel() != null ? request.getModel() : "",
                escapeJson(content),
                tokens, duration, tps);
        System.out.println(json);
    }

    public SessionStats getSessionStats() {
        var stats = sdkSession != null ? sdkSession.getStats() : null;
        if (stats == null) {
            return new SessionStats(java.time.Instant.now(), 0, 0, 0, 0, 0, 0, 0, 0, java.util.Map.of(), java.util.Map.of());
        }
        return new SessionStats(
                stats.sessionStart(),
                stats.sessionDurationSeconds(),
                stats.totalRequests(),
                stats.totalTokens(),
                stats.totalDurationMs(),
                stats.totalErrors(),
                0, 0, 0, // Placeholder for TTFT, TPOT, ITL
                stats.perModelStats(),
                stats.perProviderStats()
        );
    }

    public ExecutionDiagnostics getExecutionDiagnostics() {
        ProviderRegistry registry = providerRegistry();
        return new ExecutionDiagnostics(
                lastExecutionRoute,
                lastProviderDescriptor,
                lastExecutionError,
                lastExecutionMetadata,
                registry != null,
                registry != null && providerId != null && !providerId.isBlank() && registry.hasProvider(providerId));
    }

    public record SessionStats(
            java.time.Instant sessionStart,
            long sessionDurationSeconds,
            int totalRequests,
            int totalTokens,
            long totalDurationMs,
            int totalErrors,
            long avgTtftMs,
            long avgTpotMs,
            long avgItlMs,
            java.util.Map<String, int[]> perModelStats,
            java.util.Map<String, int[]> perProviderStats
    ) {
        public double avgTokensPerRequest() {
            return totalRequests == 0 ? 0 : (double) totalTokens / totalRequests;
        }

        public double avgTokensPerSecond() {
            return totalDurationMs == 0 ? 0 : (totalTokens / (totalDurationMs / 1000.0));
        }
    }

    public record ExecutionDiagnostics(
            String route,
            String providerDescriptor,
            String lastError,
            Map<String, Object> metadata,
            boolean providerRegistryAvailable,
            boolean providerRegistered
    ) {
    }

    private String escapeJson(String val) {
        if (val == null)
            return "";
        return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }

    private boolean shouldUseDirectProviderPath(InferenceRequest request) {
        if (providerId == null || providerId.isBlank()) {
            return false;
        }
        if (sdk.isCloudProvider(providerId) || sdk.isMcpProvider(providerId)) {
            return false;
        }
        ProviderRegistry registry = providerRegistry();
        return registry != null && registry.hasProvider(providerId);
    }

    private ProviderRequest buildDirectProviderRequest(InferenceRequest request, boolean streaming) {
        InferenceRequest sessionBoundRequest = ensureSessionBinding(request);
        Object modelPathParam = request.getParameters().get("model_path");
        String providerModel = (modelPathParam != null && !String.valueOf(modelPathParam).isBlank())
                ? String.valueOf(modelPathParam)
                : sessionBoundRequest.getModel();
        return ProviderRequests.fromInferenceRequest(
                sessionBoundRequest,
                providerModel,
                streaming,
                Duration.ofSeconds(120),
                providerId,
                Map.of(),
                Map.of(
                        "request_id", sessionBoundRequest.getRequestId(),
                        "tenantId", "community",
                        "execution_entrypoint", "cli_chat_direct",
                        "chat_session_id", sessionBoundRequest.getSessionId(),
                        "persistent_session_enabled", sessionEnabledForExecution()));
    }

    private LLMProvider requireProvider() {
        ProviderRegistry registry = providerRegistry();
        if (registry == null || providerId == null) {
            throw new IllegalStateException("Provider registry is not available for local chat execution");
        }
        return registry.getProvider(providerId)
                .orElseThrow(() -> new IllegalStateException("Provider not available: " + providerId));
    }

    private StreamingProvider requireStreamingProvider() {
        LLMProvider provider = requireProvider();
        if (!(provider instanceof StreamingProvider streamingProvider)) {
            throw new IllegalStateException("Provider does not support streaming: " + providerId);
        }
        return streamingProvider;
    }

    private ChatSessionImpl activeSession() {
        if (sdkSession instanceof ChatSessionImpl impl) {
            return impl;
        }
        throw new IllegalStateException("Chat session is not using the CLI-managed ChatSessionImpl");
    }

    private boolean sessionEnabledForExecution() {
        if (sdkSession instanceof ChatSessionImpl impl) {
            return impl.isSessionEnabled();
        }
        return enableSession;
    }

    private InferenceRequest ensureSessionBinding(InferenceRequest request) {
        boolean sessionEnabled = sessionEnabledForExecution();
        String effectiveSessionId = request.getSessionId()
                .filter(id -> !id.isBlank())
                .orElse(sessionEnabled ? sdkSession.getSessionId() : null);

        Map<String, Object> params = new HashMap<>(request.getParameters());
        params.put("chat_session_enabled", sessionEnabled);
        if (effectiveSessionId != null && !effectiveSessionId.isBlank()) {
            params.put("session_id", effectiveSessionId);
        } else {
            params.remove("session_id");
        }

        boolean metadataMatches = Objects.equals(params, request.getParameters());
        boolean sessionMatches = Objects.equals(effectiveSessionId, request.getSessionId().orElse(null));
        if (metadataMatches && sessionMatches) {
            return request;
        }

        InferenceRequest.Builder builder = request.toBuilder()
                .parameters(params);
        if (effectiveSessionId != null && !effectiveSessionId.isBlank()) {
            builder.sessionId(effectiveSessionId);
        }
        return builder.build();
    }

    private SdkException directProviderFailure(String mode, RuntimeException error) {
        String providerDescriptor = describeDirectProvider();
        String summary = summarizeThrowable(error);
        return new SdkException(
                "Local inference failed via " + providerDescriptor + " during " + mode + ": " + summary,
                error);
    }

    private String describeDirectProvider() {
        ProviderRegistry registry = providerRegistry();
        if (registry == null) {
            return providerId != null ? providerId + " [provider-registry-unavailable]" : "unknown-provider";
        }
        if (providerId == null || providerId.isBlank()) {
            return "unknown-provider";
        }
        return registry.getProvider(providerId)
                .map(provider -> providerId + " [" + provider.getClass().getName() + "]")
                .orElse(providerId + " [not-registered]");
    }

    private String summarizeThrowable(Throwable error) {
        if (error == null) {
            return "unknown error";
        }
        StringBuilder summary = new StringBuilder();
        Throwable current = error;
        int depth = 0;
        while (current != null && depth < 4) {
            if (depth > 0) {
                summary.append(" <- ");
            }
            summary.append(current.getClass().getSimpleName());
            String message = current.getMessage();
            if (message != null && !message.isBlank()) {
                summary.append(": ").append(message.replace('\n', ' ').replace('\r', ' '));
            }
            current = current.getCause();
            depth++;
        }
        return summary.toString();
    }

    private void recordExecutionSnapshot(String route, String providerDescriptor, Map<String, Object> metadata) {
        this.lastExecutionRoute = route != null ? route : "unknown";
        this.lastProviderDescriptor = providerDescriptor != null ? providerDescriptor : "unknown";
        this.lastExecutionError = null;
        this.lastExecutionMetadata = metadata == null || metadata.isEmpty() ? Map.of() : Map.copyOf(metadata);
    }

    private void recordExecutionFailure(String route, String providerDescriptor, String errorSummary) {
        this.lastExecutionRoute = route != null ? route : "unknown";
        this.lastProviderDescriptor = providerDescriptor != null ? providerDescriptor : "unknown";
        this.lastExecutionError = errorSummary;
    }

    private ProviderRegistry providerRegistry() {
        if (providerRegistry != null) {
            return providerRegistry;
        }
        try {
            if (Arc.container() == null) {
                return null;
            }
            var instance = Arc.container().instance(ProviderRegistry.class);
            if (instance.isAvailable()) {
                providerRegistry = instance.get();
            }
        } catch (Exception ignored) {
            // Fall through and let callers handle the missing registry path.
        }
        return providerRegistry;
    }
}
