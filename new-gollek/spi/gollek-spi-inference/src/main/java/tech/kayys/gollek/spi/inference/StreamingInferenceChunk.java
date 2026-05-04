package tech.kayys.gollek.spi.inference;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import tech.kayys.gollek.spi.model.ModalityType;

import java.time.Instant;

import java.util.Map;

/**
 * A single incremental chunk emitted during a streaming multimodal inference call.
 *
 * <p>Chunks are delivered over Server-Sent Events (SSE) on
 * {@code POST /v1/multimodal/stream}.  Consumers reconstruct the full response
 * by accumulating delta values until a chunk with {@code finished=true} arrives.
 *
 * <h3>SSE wire format</h3>
 * <pre>
 *   data: {"requestId":"abc","index":0,"modality":"TEXT","delta":"Hello","finished":false}
 *   data: {"requestId":"abc","index":1,"modality":"TEXT","delta":" world","finished":false}
 *   data: {"requestId":"abc","index":2,"modality":"TEXT","delta":"!","finished":true,
 *          "finishReason":"stop","usage":{"inputTokens":20,"outputTokens":3,"latencyMs":412}}
 * </pre>
 * 
 *  * <h2>Example Usage</h2>
 * 
 * <pre>{@code
 * Multi<StreamingInferenceChunk> stream = engine.stream(request);
 * stream.subscribe(chunk -> {
 *     System.out.print(chunk.delta());
 *     if (chunk.isComplete()) {
 *         System.out.println("\nStreaming complete!");
 *     }
 * });
 * }</pre>
 */
@JsonInclude(Include.NON_NULL)
public record StreamingInferenceChunk(

        /** Echoed from the original request for client-side multiplexing. */
        String       requestId,

        /** Zero-based sequence number; clients can detect dropped chunks. */
        int          index,

        /** The modality of this delta (TEXT for token-by-token, IMAGE for partial image). */
        ModalityType modality,

        /** Incremental text delta (non-null when modality is TEXT). */
        String       delta,

        /** Partial base64-encoded image chunk (non-null when modality is IMAGE). */
        String       imageDeltaBase64,

        /** True on the final chunk of the stream. */
        boolean      finished,

        /** Why the stream ended: "stop" | "length" | "error" | "content_filter". */
        String       finishReason,

        /** Usage statistics included only on the final chunk. */
        ChunkUsage   usage,

        /** Server-side timestamp of this chunk. */
        Instant      emittedAt,

        /** Optional metadata associated with this chunk (e.g., hardware execution info). */
        Map<String, Object> metadata
) {
    /** Alias for index() for backwards compatibility. */
    public int getIndex() { return index(); }

    /** Alias for delta() for backwards compatibility. */
    public String getDelta() { return delta(); }

    // -------------------------------------------------------------------------
    // Factory helpers
    // -------------------------------------------------------------------------

    public static StreamingInferenceChunk textDelta(String requestId, int index, String delta) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT,
                delta, null, false, null, null, Instant.now(), null);
    }

    public static StreamingInferenceChunk finalTextChunk(String requestId, int index,
                                                 String delta, ChunkUsage usage) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT,
                delta, null, true, "stop", usage, Instant.now(), null);
    }

    public static StreamingInferenceChunk errorChunk(String requestId, int index, String message) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT,
                message, null, true, "error", null, Instant.now(), null);
    }

    public static StreamingInferenceChunk imageChunk(String requestId, int index,
                                             String base64Delta, boolean finished) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.IMAGE,
                null, base64Delta, finished, finished ? "stop" : null, null, Instant.now(), null);
    }

        /**
     * Create a non-final chunk.
     */
    public static StreamingInferenceChunk of(String requestId, int index, String delta) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT, delta, null, false, null, null, Instant.now(), null);
    }

    /**
     * Create a chunk with metadata.
     */
    public static StreamingInferenceChunk withMetadata(String requestId, int index, String delta, Map<String, Object> metadata) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT, delta, null, false, null, null, Instant.now(), metadata);
    }

    /**
     * Create the final chunk.
     */
    public static StreamingInferenceChunk finalChunk(String requestId, int index, String delta) {
        return new StreamingInferenceChunk(requestId, index, ModalityType.TEXT, delta, null, true, "stop", null, Instant.now(), null);
    }

    // -------------------------------------------------------------------------
    // Nested usage record
    // -------------------------------------------------------------------------

    public record ChunkUsage(long inputTokens, long outputTokens, long latencyMs) {}
}

