package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.provider.ProviderRequest;

/**
 * GGUF backend powered by the existing llama.cpp provider.
 */
final class LlamaCppGgufBackend implements GgufBackend {
    private final Object provider;

    LlamaCppGgufBackend(Object provider) {
        this.provider = provider;
    }

    @Override
    public String name() {
        return "llamacpp";
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        if (request.getInferenceRequest().isEmpty()) {
            return RunnerResult.failed("Unsupported request type for llama.cpp GGUF backend");
        }

        InferenceRequest inferenceRequest = request.getInferenceRequest().get();
        ProviderRequest providerRequest = ProviderRequest.builder()
                .requestId(inferenceRequest.getRequestId())
                .model(inferenceRequest.getModel())
                .messages(inferenceRequest.getMessages())
                .parameters(inferenceRequest.getParameters())
                .tools(inferenceRequest.getTools())
                .toolChoice(inferenceRequest.getToolChoice())
                .streaming(inferenceRequest.isStreaming())
                .timeout(inferenceRequest.getTimeout().orElse(java.time.Duration.ofSeconds(30)))
                .userId(inferenceRequest.getUserId().orElse(null))
                .sessionId(inferenceRequest.getSessionId().orElse(null))
                .traceId(inferenceRequest.getTraceId().orElse(null))
                .apiKey(inferenceRequest.getApiKey())
                .metadata(inferenceRequest.getMetadata())
                .preferredProvider(inferenceRequest.getPreferredProvider().orElse(null))
                .nativeContextSegment(inferenceRequest.getNativeContextSegment().orElse(null))
                .build();

        try {
            Object uni = provider.getClass()
                    .getMethod("infer", ProviderRequest.class)
                    .invoke(provider, providerRequest);
            Object awaiter = uni.getClass().getMethod("await").invoke(uni);
            Object response = awaiter.getClass().getMethod("indefinitely").invoke(awaiter);
            return (RunnerResult<T>) RunnerResult.success(response);
        } catch (Exception e) {
            return RunnerResult.failed("Llama.cpp GGUF inference failed: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        // LlamaCppProvider lifecycle is managed by CDI or the embedding application.
    }
}
