package tech.kayys.gollek.multimodal;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.model.MultimodalCapability;
import tech.kayys.gollek.spi.model.MultimodalContent;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;

/**
 * Service Provider Interface for multimodal inference backends.
 *
 * <p>Implement this interface to integrate a new multimodal backend
 * (e.g. Google Gemini, OpenAI GPT-4o, LLaVA via llama.cpp, Whisper, etc.)
 * into the Gollek plugin system.
 *
 * <h3>Implementation contract</h3>
 * <ul>
 *   <li>Implementations MUST be annotated with {@code @ApplicationScoped} (CDI).</li>
 *   <li>{@link #capability()} is called once at startup for capability registration.</li>
 *   <li>{@link #infer(MultimodalRequest)} is the primary non-streaming entry-point.</li>
 *   <li>{@link #inferStream(MultimodalRequest)} is required only when
 *       {@link MultimodalCapability#isSupportsStreaming()} returns {@code true}.</li>
 *   <li>Implementations MUST propagate the request {@code requestId} to the response.</li>
 *   <li>Errors MUST be returned as {@link MultimodalResponse} with status {@code ERROR};
 *       never throw unchecked exceptions from the Uni/Multi pipelines.</li>
 * </ul>
 *
 * <h3>Minimal skeleton</h3>
 * <pre>{@code
 * @ApplicationScoped
 * public class MyVisionProvider implements MultimodalInferenceProvider {
 *
 *     public String providerId()      { return "my-vision"; }
 *     public boolean isAvailable()    { return true; }
 *
 *     public MultimodalCapability capability() {
 *         return MultimodalCapability.builder("my-model")
 *             .inputModalities(TEXT, IMAGE)
 *             .outputModalities(TEXT)
 *             .build();
 *     }
 *
 *     public Uni<MultimodalResponse> infer(MultimodalRequest req) {
 *         return Uni.createFrom().item(() -> callMyBackend(req));
 *     }
 * }
 * }</pre>
 */
public interface MultimodalInferenceProvider {

    /**
     * Unique identifier for this provider, e.g. {@code "openai-gpt4o"},
     * {@code "gemini-pro-vision"}, {@code "llava-13b-gguf"}.
     */
    String providerId();

    /**
     * Returns {@code true} when the backend is reachable and ready to serve.
     * Called periodically by the health-check subsystem.
     */
    boolean isAvailable();

    /**
     * Describes the full multimodal contract of this provider/model.
     * Called once at startup; the result is cached by the capability registry.
     */
    MultimodalCapability capability();

    // -------------------------------------------------------------------------
    // Core inference
    // -------------------------------------------------------------------------

    /**
     * Perform a complete (non-streaming) multimodal inference call.
     *
     * @param request the validated multimodal request
     * @return a {@link Uni} that emits exactly one {@link MultimodalResponse}
     */
    Uni<MultimodalResponse> infer(MultimodalRequest request);

    /**
     * Perform a streaming multimodal inference call.
     *
     * <p>Default implementation wraps {@link #infer(MultimodalRequest)} as a
     * single-item stream.  Override for true token/chunk streaming.
     *
     * @param request the validated multimodal request
     * @return a {@link Multi} of incremental {@link MultimodalContent} deltas
     */
    default Multi<MultimodalContent> inferStream(MultimodalRequest request) {
        return infer(request)
                .onItem().transformToMulti(resp ->
                        Multi.createFrom().items(resp.getOutputs()));
    }

    // -------------------------------------------------------------------------
    // Lifecycle hooks (optional)
    // -------------------------------------------------------------------------

    /**
     * Called once after CDI initialization.  Override for lazy model loading,
     * connection pool warm-up, etc.
     */
    default void onStart() {}

    /**
     * Called during graceful shutdown.  Override to close connections, release
     * GPU memory, etc.
     */
    default void onStop() {}
}
