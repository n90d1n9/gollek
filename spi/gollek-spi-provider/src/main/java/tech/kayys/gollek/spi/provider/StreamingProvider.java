package tech.kayys.gollek.spi.provider;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

/**
 * Extension for providers that support streaming output.
 */
public interface StreamingProvider extends LLMProvider {

    /**
     * Execute streaming inference request.
     * 
     * @param request Normalized inference request
     * @param context Tenant context
     * @return Multi with stream chunks
     */
    Multi<StreamingInferenceChunk> inferStream(ProviderRequest request);
}
