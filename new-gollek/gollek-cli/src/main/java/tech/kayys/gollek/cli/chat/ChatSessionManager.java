package tech.kayys.gollek.cli.chat;

import jakarta.enterprise.context.Dependent;
import jakarta.inject.Inject;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.exception.SdkException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.sdk.session.ChatSession;
import tech.kayys.gollek.sdk.session.ChatSessionFactory;

import java.io.PrintWriter;
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
    
    private String modelId;
    private String providerId;
    private String modelPathOverride;

    // UI/Output hooks
    private ChatUIRenderer uiRenderer;
    private PrintWriter fileWriter;
    private boolean quiet;
    private boolean autoContinue = true;
    private int maxTokens = 256;
    private double temperature = 0.2;

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
        
        this.sdkSession = sessionFactory.createSession(modelId, providerId);
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
        this.sdkSession = sessionFactory.createSession(newModelId, providerId);
    }

    public String getModelId() {
        return modelId;
    }

    public String getProviderId() {
        return providerId;
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

        try {
            if (stream) {
                executeStreaming(request, enableJsonSse);
            } else {
                executeNonStreaming(request);
            }
        } catch (SdkException e) {
            uiRenderer.printError("Inference failed: " + e.getMessage(), quiet);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            uiRenderer.printError("Inference interrupted", quiet);
        }
    }

    private void executeStreaming(InferenceRequest request, boolean enableJsonSse) throws InterruptedException, SdkException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicInteger tokenCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        AtomicLong firstTokenTime = new AtomicLong(0);
        AtomicLong lastTokenTime = new AtomicLong(0);
        AtomicLong sumItl = new AtomicLong(0);
        StringBuilder fullResponse = new StringBuilder();

        boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

        CliSpinner spinner = new CliSpinner(System.out, "Thinking…");
        if (!quiet && !enableJsonSse && !jsonMode) {
            spinner.start();
        }

        sdkSession.stream(request)
                .subscribe().with(
                        chunk -> {
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

                            String delta = chunk.getDelta();
                            if (delta == null) return;
                            
                            fullResponse.append(delta);
                            tokenCount.incrementAndGet();
                            
                            if (delta.isEmpty()) return;
                            
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
                            uiRenderer.printError("Stream error: " + error.getMessage(), quiet);
                            latch.countDown();
                        },
                        () -> {
                            spinner.stop();
                            long duration = System.currentTimeMillis() - startTime;
                            double tps = (tokenCount.get() / (Math.max(1, duration) / 1000.0));
                            
                            if (enableJsonSse) {
                                printOpenAiSseFinal(request.getRequestId(), request.getModel());
                            } else if (jsonMode) {
                                System.out.println();
                                printJsonModeResponse(request, fullResponse.toString(), tokenCount.get(), duration / 1000.0, tps);
                            } else {
                                System.out.println();
                                uiRenderer.printStats(tokenCount.get(), duration / 1000.0, tps, quiet);
                            }

                            latch.countDown();
                        });

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
            
            boolean jsonMode = request.getParameters().getOrDefault("json_mode", false) instanceof Boolean jm && jm;

            if (jsonMode) {
                printJsonModeResponse(request, response.getContent(), response.getTokensUsed(), duration / 1000.0, tps);
            } else {
                uiRenderer.printAssistantPrefix(quiet, false);
                System.out.println(response.getContent());
                uiRenderer.printStats(response.getTokensUsed(), duration / 1000.0, tps, quiet);
            }
        } finally {
            spinner.stop();
        }
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

    private String escapeJson(String val) {
        if (val == null)
            return "";
        return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r");
    }
}
