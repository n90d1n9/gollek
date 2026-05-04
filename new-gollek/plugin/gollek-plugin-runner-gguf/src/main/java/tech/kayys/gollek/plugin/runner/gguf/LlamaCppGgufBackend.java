package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.inference.llamacpp.LlamaCppProvider;
import tech.kayys.gollek.plugin.runner.RunnerRequest;
import tech.kayys.gollek.plugin.runner.RunnerResult;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import java.util.UUID;

/**
 * GGUF backend powered by llama.cpp native bindings.
 */
public class LlamaCppGgufBackend implements GgufBackend {
    
    private final LlamaCppProvider provider;
    
    public LlamaCppGgufBackend(LlamaCppProvider provider) {
        this.provider = provider;
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> RunnerResult<T> execute(RunnerRequest request) {
        if (request.getInferenceRequest().isPresent()) {
            tech.kayys.gollek.spi.inference.InferenceRequest req = request.getInferenceRequest().get();
            
            // Bridge InferenceRequest to ProviderRequest
            ProviderRequest providerRequest = ProviderRequest.builder()
                    .requestId(req.getRequestId())
                    .model(req.getModel())
                    .messages(req.getMessages())
                    .parameters(req.getParameters())
                    .tools(req.getTools())
                    .toolChoice(req.getToolChoice())
                    .streaming(req.isStreaming())
                    .timeout(req.getTimeout().orElse(java.time.Duration.ofSeconds(30)))
                    .userId(req.getUserId().orElse(null))
                    .sessionId(req.getSessionId().orElse(null))
                    .traceId(req.getTraceId().orElse(null))
                    .apiKey(req.getApiKey())
                    .metadata(req.getMetadata())
                    .preferredProvider(req.getPreferredProvider().orElse(null))
                    .build();

            try {
                InferenceResponse response = provider.infer(providerRequest).await().indefinitely();
                return (RunnerResult<T>) RunnerResult.success(response);
            } catch (Exception e) {
                return RunnerResult.failed("Llama.cpp inference failed: " + e.getMessage());
            }
        }
        return RunnerResult.failed("Unsupported request type for llama.cpp backend");
    }

    @Override
    public void close() {
        // LlamaCppProvider lifecycle is managed by CDI
    }
}
