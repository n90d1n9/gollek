package tech.kayys.gollek.plugin.runner.gguf;

import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;

import java.util.Map;
import java.util.UUID;

/**
 * Session for GGUF model execution.
 */
public interface GgufRunnerSession extends RunnerSession {
    
    @Override
    default String getSessionId() {
        return UUID.randomUUID().toString();
    }

    Uni<InferenceResponse> infer(InferenceRequest request);
    
    Multi<InferenceResponse> inferStream(InferenceRequest request);
}
