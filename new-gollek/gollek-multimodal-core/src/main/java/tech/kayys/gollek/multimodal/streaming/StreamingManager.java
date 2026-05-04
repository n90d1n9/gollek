package tech.kayys.gollek.multimodal.streaming;

import io.smallrye.mutiny.Multi;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.spi.processor.MultimodalProcessor;

import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Streaming manager for multimodal inference.
 * Handles token-by-token streaming generation.
 */
public class StreamingManager {

    private static final Logger log = Logger.getLogger(StreamingManager.class);
    private static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(5);

    private final Map<String, StreamingState> activeStreams = new ConcurrentHashMap<>();
    private final ExecutorService executorService;

    public StreamingManager() {
        this.executorService = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "multimodal-streaming-worker");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Create a streaming response from a processor.
     */
    public Multi<MultimodalResponse> createStream(
            MultimodalProcessor processor,
            MultimodalRequest request) {

        return Multi.createFrom().emitter(emitter -> {
            final String requestId = (request.getRequestId() == null || request.getRequestId().isBlank())
                    ? "stream-" + System.currentTimeMillis()
                    : request.getRequestId();

            // Create streaming state
            StreamingState state = new StreamingState(requestId, request.getModel());
            activeStreams.put(requestId, state);

            log.infof("Starting multimodal stream: %s", requestId);

            // Execute in background
            executorService.submit(() -> {
                try {
                    // Process the request
                    MultimodalResponse fullResponse = processor.process(request)
                            .await().atMost(DEFAULT_TIMEOUT);

                    // Stream the response token by token
                    String fullText = fullResponse.getOutputs()[0].getText();
                    String[] tokens = tokenize(fullText);

                    for (String token : tokens) {
                        if (state.isCancelled()) {
                            log.infof("Stream cancelled: %s", requestId);
                            break;
                        }

                        // Create streaming chunk
                        MultimodalResponse chunk = MultimodalResponse.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .outputs(MultimodalContent.ofText(token))
                                .status(MultimodalResponse.ResponseStatus.IN_PROGRESS)
                                .metadata(Map.of(
                                        "stream", "true",
                                        "token_index", String.valueOf(state.getTokensGenerated())))
                                .build();

                        emitter.emit(chunk);
                        state.recordToken(token);
                    }

                    // Send final response
                    if (!state.isCancelled()) {
                        MultimodalResponse finalResponse = MultimodalResponse.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .outputs(fullResponse.getOutputs())
                                .status(MultimodalResponse.ResponseStatus.SUCCESS)
                                .usage(fullResponse.getUsage())
                                .durationMs(fullResponse.getDurationMs())
                                .metadata(Map.of(
                                        "stream", "true",
                                        "complete", "true",
                                        "total_tokens", String.valueOf(state.getTokensGenerated())))
                                .build();

                        emitter.emit(finalResponse);
                        state.markComplete();
                        log.infof("Stream completed: %s - %s", requestId, state.getStats());
                    }

                    emitter.complete();

                } catch (Exception e) {
                    log.errorf("Stream error: %s - %s", requestId, e.getMessage());

                    // Send error response
                    MultimodalResponse errorResponse = MultimodalResponse.builder()
                            .requestId(requestId)
                            .model(request.getModel())
                            .status(MultimodalResponse.ResponseStatus.ERROR)
                            .metadata(Map.of(
                                    "error", e.getMessage(),
                                    "stream", "true"))
                            .build();

                    emitter.emit(errorResponse);
                    emitter.fail(e);
                    state.cancel();
                } finally {
                    activeStreams.remove(requestId);
                }
            });

            // Handle cancellation
            emitter.onTermination(() -> {
                StreamingState activeState = activeStreams.get(requestId);
                if (activeState != null) {
                    activeState.cancel();
                    log.infof("Stream terminated: %s", requestId);
                }
            });

        });
    }

    /**
     * Create a streaming response with custom tokenization.
     */
    public Multi<MultimodalResponse> createStream(
            MultimodalProcessor processor,
            MultimodalRequest request,
            Tokenizer tokenizer) {

        return Multi.createFrom().emitter(emitter -> {
            final String requestId = (request.getRequestId() == null || request.getRequestId().isBlank())
                    ? "stream-" + System.currentTimeMillis()
                    : request.getRequestId();

            StreamingState state = new StreamingState(requestId, request.getModel());
            activeStreams.put(requestId, state);

            executorService.submit(() -> {
                try {
                    // Get full response
                    MultimodalResponse fullResponse = processor.process(request)
                            .await().atMost(DEFAULT_TIMEOUT);

                    // Stream using custom tokenizer
                    String fullText = fullResponse.getOutputs()[0].getText();
                    Iterable<String> tokens = tokenizer.tokenize(fullText);

                    for (String token : tokens) {
                        if (state.isCancelled()) {
                            break;
                        }

                        MultimodalResponse chunk = MultimodalResponse.builder()
                                .requestId(requestId)
                                .model(request.getModel())
                                .outputs(MultimodalContent.ofText(token))
                                .status(MultimodalResponse.ResponseStatus.IN_PROGRESS)
                                .build();

                        emitter.emit(chunk);
                        state.recordToken(token);
                    }

                    // Send final response
                    MultimodalResponse finalResponse = MultimodalResponse.builder()
                            .requestId(requestId)
                            .model(request.getModel())
                            .outputs(fullResponse.getOutputs())
                            .status(MultimodalResponse.ResponseStatus.SUCCESS)
                            .usage(fullResponse.getUsage())
                            .durationMs(fullResponse.getDurationMs())
                            .build();

                    emitter.emit(finalResponse);
                    state.markComplete();
                    emitter.complete();

                } catch (Exception e) {
                    emitter.fail(e);
                    state.cancel();
                } finally {
                    activeStreams.remove(requestId);
                }
            });

            // Handle cancellation
            emitter.onTermination(() -> {
                StreamingState activeState = activeStreams.get(requestId);
                if (activeState != null) {
                    activeState.cancel();
                    log.infof("Stream terminated: %s", requestId);
                }
            });

        });
    }

    /**
     * Get active stream count.
     */
    public int getActiveStreamCount() {
        return activeStreams.size();
    }

    /**
     * Get streaming state for a request.
     */
    public StreamingState getStreamingState(String requestId) {
        return activeStreams.get(requestId);
    }

    /**
     * Cancel an active stream.
     */
    public boolean cancelStream(String requestId) {
        StreamingState state = activeStreams.get(requestId);
        if (state != null) {
            state.cancel();
            log.infof("Stream cancelled: %s", requestId);
            return true;
        }
        return false;
    }

    /**
     * Get streaming statistics for all active streams.
     */
    public Map<String, StreamingState.StreamingStats> getAllStreamStats() {
        return activeStreams.values().stream()
                .collect(
                        java.util.stream.Collectors.toMap(
                                StreamingState::getRequestId,
                                StreamingState::getStats));
    }

    /**
     * Simple word-level tokenizer.
     */
    private String[] tokenize(String text) {
        // Split on spaces but keep punctuation
        return text.split("(?<=\\s)|(?=\\s)");
    }

    /**
     * Tokenizer interface for custom tokenization.
     */
    public interface Tokenizer {
        Iterable<String> tokenize(String text);
    }

    /**
     * Shutdown the streaming manager.
     */
    public void shutdown() {
        log.info("Shutting down streaming manager");

        // Cancel all active streams
        activeStreams.values().forEach(StreamingState::cancel);
        activeStreams.clear();

        executorService.shutdown();
    }
}
