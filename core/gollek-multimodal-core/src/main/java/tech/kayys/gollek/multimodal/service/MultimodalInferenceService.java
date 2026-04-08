package tech.kayys.gollek.multimodal.service;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.batch.BatchProcessor;
import tech.kayys.gollek.spi.model.MultimodalRequest;
import tech.kayys.gollek.spi.model.MultimodalResponse;
import tech.kayys.gollek.spi.processor.MultimodalProcessor;
import tech.kayys.gollek.multimodal.streaming.StreamingManager;
import tech.kayys.gollek.multimodal.streaming.StreamingState;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Multimodal inference service.
 * Integrates with the Gollek inference engine.
 */
@ApplicationScoped
public class MultimodalInferenceService {

    private static final Logger log = Logger.getLogger(MultimodalInferenceService.class);

    @Inject
    jakarta.enterprise.inject.Instance<MultimodalProcessor> processors;

    private final Map<String, MultimodalProcessor> processorRegistry = new ConcurrentHashMap<>();
    private final StreamingManager streamingManager;
    private final BatchProcessor batchProcessor;

    /**
     * Initialize the service.
     */
    public MultimodalInferenceService() {
        this.streamingManager = new StreamingManager();
        this.batchProcessor = new BatchProcessor(32, Duration.ofMillis(100), Duration.ofMinutes(2));
    }

    /**
     * Initialize the service.
     */
    public void initialize() {
        log.info("Initializing Multimodal Inference Service");
        
        // Register all discovered processors
        for (MultimodalProcessor processor : processors) {
            processorRegistry.put(processor.getProcessorId(), processor);
            log.infof("Registered multimodal processor: %s", processor.getProcessorId());
        }
        
        // Start batch processor
        if (!processorRegistry.isEmpty()) {
            batchProcessor.setProcessor(processorRegistry.values().iterator().next());
            batchProcessor.start();
        }
        
        log.infof("Multimodal Inference Service initialized with %d processors", 
                  processorRegistry.size());
    }

    /**
     * Process a multimodal request.
     *
     * @param request the multimodal request
     * @return Uni containing the response
     */
    public Uni<MultimodalResponse> infer(MultimodalRequest request) {
        log.infof("Processing multimodal request: %s", request.getRequestId());
        
        // Select appropriate processor based on model
        MultimodalProcessor processor = selectProcessor(request);
        
        if (processor == null) {
            return Uni.createFrom().failure(
                new IllegalArgumentException("No processor available for model: " + request.getModel())
            );
        }
        
        if (!processor.isAvailable()) {
            return Uni.createFrom().failure(
                new IllegalStateException("Processor " + processor.getProcessorId() + " is not available")
            );
        }
        
        // Check if batch processing is requested
        if (shouldUseBatch(request)) {
            return batchProcessor.submit(request);
        }
        
        // Process the request
        return processor.process(request)
            .onItem().invoke(response -> 
                log.infof("Multimodal inference completed: %s in %dms", 
                         request.getRequestId(), response.getDurationMs())
            );
    }

    /**
     * Process a multimodal request with streaming.
     *
     * @param request the multimodal request
     * @return Multi containing streaming responses
     */
    public Multi<MultimodalResponse> inferStream(MultimodalRequest request) {
        log.infof("Processing streaming multimodal request: %s", request.getRequestId());
        
        MultimodalProcessor processor = selectProcessor(request);
        
        if (processor == null) {
            return Multi.createFrom().failure(
                new IllegalArgumentException("No processor available for model: " + request.getModel())
            );
        }
        
        return streamingManager.createStream(processor, request);
    }

    /**
     * Process a batch of requests.
     *
     * @param requests the multimodal requests
     * @return Uni containing list of responses
     */
    public Uni<List<MultimodalResponse>> inferBatch(List<MultimodalRequest> requests) {
        log.infof("Processing batch of %d requests", requests.size());
        
        // Submit all requests
        List<Uni<MultimodalResponse>> responses = requests.stream()
            .map(this::infer)
            .collect(java.util.stream.Collectors.toList());
        
        // Collect all responses
        return Uni.combine().all().unis(responses)
            .with(list -> {
                java.util.List<MultimodalResponse> result = new java.util.ArrayList<>();
                for (Object o : list) {
                    result.add((MultimodalResponse) o);
                }
                return result;
            });
    }

    /**
     * Cancel an active streaming request.
     */
    public boolean cancelStream(String requestId) {
        return streamingManager.cancelStream(requestId);
    }

    /**
     * Get streaming statistics.
     */
    public Map<String, StreamingState.StreamingStats> getActiveStreams() {
        return streamingManager.getAllStreamStats();
    }

    /**
     * Get active stream count.
     */
    public int getActiveStreamCount() {
        return streamingManager.getActiveStreamCount();
    }

    /**
     * Get batch processing statistics.
     */
    public BatchProcessor.BatchStats getBatchStats() {
        return batchProcessor.getStats();
    }

    /**
     * Get pending batch count.
     */
    public int getPendingBatchCount() {
        return batchProcessor.getPendingCount();
    }

    /**
     * Select the appropriate processor for the request.
     *
     * @param request the multimodal request
     * @return selected processor or null
     */
    private MultimodalProcessor selectProcessor(MultimodalRequest request) {
        String model = request.getModel();
        
        // Try to find processor by model name pattern
        for (MultimodalProcessor processor : processorRegistry.values()) {
            if (processor.isAvailable() && supportsModel(processor, model)) {
                return processor;
            }
        }
        
        // Fallback to first available processor
        return processorRegistry.values().stream()
            .filter(MultimodalProcessor::isAvailable)
            .findFirst()
            .orElse(null);
    }

    /**
     * Check if a processor supports a model.
     *
     * @param processor the processor
     * @param model the model name
     * @return true if supported
     */
    private boolean supportsModel(MultimodalProcessor processor, String model) {
        // Simple pattern matching - can be enhanced with capability registry
        String processorId = processor.getProcessorId().toLowerCase();
        String modelLower = model.toLowerCase();
        
        return processorId.contains(modelLower) || modelLower.contains(processorId);
    }

    /**
     * Check if batch processing should be used.
     */
    private boolean shouldUseBatch(MultimodalRequest request) {
        // Use batch processing if requested or if queue is not empty
        if (request.getParameters() != null && 
            request.getParameters().containsKey("use_batch")) {
            Object useBatch = request.getParameters().get("use_batch");
            return useBatch instanceof Boolean && (Boolean) useBatch;
        }
        
        // Auto-use batch if queue has pending requests
        return batchProcessor.getPendingCount() > 0;
    }

    /**
     * Get all registered processors.
     *
     * @return list of processor IDs
     */
    public List<String> getAvailableProcessors() {
        return processorRegistry.keySet().stream().toList();
    }

    /**
     * Get processor by ID.
     *
     * @param processorId the processor ID
     * @return the processor or null
     */
    public MultimodalProcessor getProcessor(String processorId) {
        return processorRegistry.get(processorId);
    }

    /**
     * Shutdown the service.
     */
    public void shutdown() {
        log.info("Shutting down Multimodal Inference Service");
        streamingManager.shutdown();
        batchProcessor.stop();
        processorRegistry.clear();
    }
}
