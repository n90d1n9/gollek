package tech.kayys.gollek.multimodal.streaming;

/**
 * Streaming state for multimodal inference.
 * Tracks the current state of a streaming generation.
 */
public class StreamingState {

    private final String requestId;
    private final String model;
    private final long startTime;
    private volatile boolean complete;
    private volatile boolean cancelled;
    private int tokensGenerated;
    private long lastTokenTime;
    private StringBuilder accumulatedText;

    public StreamingState(String requestId, String model) {
        this.requestId = requestId;
        this.model = model;
        this.startTime = System.currentTimeMillis();
        this.complete = false;
        this.cancelled = false;
        this.tokensGenerated = 0;
        this.lastTokenTime = System.currentTimeMillis();
        this.accumulatedText = new StringBuilder();
    }

    /**
     * Record a new token in the stream.
     */
    public synchronized void recordToken(String token) {
        if (complete || cancelled) {
            return;
        }
        
        tokensGenerated++;
        lastTokenTime = System.currentTimeMillis();
        accumulatedText.append(token);
    }

    /**
     * Mark the stream as complete.
     */
    public synchronized void markComplete() {
        this.complete = true;
        this.lastTokenTime = System.currentTimeMillis();
    }

    /**
     * Cancel the stream.
     */
    public synchronized void cancel() {
        this.cancelled = true;
        this.lastTokenTime = System.currentTimeMillis();
    }

    /**
     * Get the accumulated text so far.
     */
    public synchronized String getAccumulatedText() {
        return accumulatedText.toString();
    }

    /**
     * Get tokens generated per second.
     */
    public double getTokensPerSecond() {
        long elapsed = System.currentTimeMillis() - startTime;
        if (elapsed == 0) return 0;
        return tokensGenerated / (elapsed / 1000.0);
    }

    /**
     * Get time to first token in milliseconds.
     */
    public long getTimeToFirstToken() {
        return tokensGenerated > 0 ? (lastTokenTime - startTime) : 0;
    }

    // Getters
    
    public String getRequestId() {
        return requestId;
    }

    public String getModel() {
        return model;
    }

    public long getStartTime() {
        return startTime;
    }

    public boolean isComplete() {
        return complete;
    }

    public boolean isCancelled() {
        return cancelled;
    }

    public int getTokensGenerated() {
        return tokensGenerated;
    }

    public long getLastTokenTime() {
        return lastTokenTime;
    }

    /**
     * Get streaming statistics.
     */
    public StreamingStats getStats() {
        long totalDuration = System.currentTimeMillis() - startTime;
        return new StreamingStats(
            requestId,
            model,
            tokensGenerated,
            totalDuration,
            getTokensPerSecond(),
            getTimeToFirstToken(),
            complete,
            cancelled
        );
    }

    /**
     * Streaming statistics.
     */
    public static class StreamingStats {
        public final String requestId;
        public final String model;
        public final int tokensGenerated;
        public final long totalDurationMs;
        public final double tokensPerSecond;
        public final long timeToFirstTokenMs;
        public final boolean complete;
        public final boolean cancelled;

        public StreamingStats(String requestId, String model, int tokensGenerated,
                             long totalDurationMs, double tokensPerSecond,
                             long timeToFirstTokenMs, boolean complete, boolean cancelled) {
            this.requestId = requestId;
            this.model = model;
            this.tokensGenerated = tokensGenerated;
            this.totalDurationMs = totalDurationMs;
            this.tokensPerSecond = tokensPerSecond;
            this.timeToFirstTokenMs = timeToFirstTokenMs;
            this.complete = complete;
            this.cancelled = cancelled;
        }

        @Override
        public String toString() {
            return String.format(
                "StreamingStats{requestId=%s, tokens=%d, duration=%dms, tps=%.2f, ttft=%dms, complete=%s}",
                requestId, tokensGenerated, totalDurationMs, tokensPerSecond, 
                timeToFirstTokenMs, complete
            );
        }
    }
}
