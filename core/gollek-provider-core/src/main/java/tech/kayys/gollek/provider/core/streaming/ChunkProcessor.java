package tech.kayys.gollek.provider.core.streaming;

import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

/**
 * Processes raw chunks into InferenceChunk objects
 */
public interface ChunkProcessor {

    /**
     * Process raw chunk data
     */
    StreamingInferenceChunk process(String rawChunk, String requestId, int index);

    /**
     * Check if chunk indicates end of stream
     */
    boolean isEndOfStream(String rawChunk);

    /**
     * Extract finish reason from chunk
     */
    String extractFinishReason(String rawChunk);
}