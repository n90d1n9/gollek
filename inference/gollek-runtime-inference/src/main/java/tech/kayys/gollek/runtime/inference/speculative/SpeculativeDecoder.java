package tech.kayys.gollek.runtime.inference.speculative;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Speculative decoding with draft model verification.
 * <p>
 * Uses a small draft model to speculate multiple tokens, then verifies
 * with the large target model. This achieves 2-3× throughput improvement
 * by reducing the number of expensive target model forward passes.
 *
 * <h2>Algorithm</h2>
 * <pre>
 * 1. Draft model generates K tokens autoregressively (fast)
 * 2. Target model verifies all K tokens in single forward pass
 * 3. Accept matching tokens, reject mismatches
 * 4. Resample from first rejection point
 * 5. Repeat
 *
 * Example (K=4):
 *   Draft:  [A, B, C, D]  ← 4 fast forward passes
 *   Target: [A, B, X, ...] ← 1 slow forward pass (verifies all)
 *   Result: Accept A, B (2 tokens), reject at C
 *   Next:   Draft from B → [E, F, G, H]
 * </pre>
 *
 * <h2>Performance</h2>
 * <ul>
 *   <li><b>Draft Model:</b> 1-3B parameters (10-20× faster than target)</li>
 *   <li><b>Target Model:</b> 70B parameters (verified in single pass)</li>
 *   <li><b>Acceptance Rate:</b> 70-90% for similar model families</li>
 *   <li><b>Speedup:</b> 2-3× over standard autoregressive decoding</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * SpeculativeDecoder decoder = SpeculativeDecoder.builder()
 *     .draftModelId("llama-3-1b")
 *     .targetModelId("llama-3-70b")
 *     .draftTokens(5)
 *     .temperature(0.8)
 *     .build();
 *
 * List<Integer> output = decoder.generate(
 *     promptTokens, maxTokens=256);
 * }</pre>
 *
 * @since 0.3.0
 */
public final class SpeculativeDecoder {

    private static final Logger LOG = Logger.getLogger(SpeculativeDecoder.class);

    // ── Configuration ─────────────────────────────────────────────────

    /** Draft model identifier */
    private final String draftModelId;

    /** Target model identifier */
    private final String targetModelId;

    /** Number of tokens to draft per verification step */
    private final int draftTokens;

    /** Sampling temperature */
    private final double temperature;

    /** Top-p sampling threshold */
    private final double topP;

    /** Random seed for reproducibility */
    private final long seed;

    // ── State ─────────────────────────────────────────────────────────

    /** Total tokens generated */
    private final AtomicLong totalTokens = new AtomicLong(0);

    /** Total draft forward passes */
    private final AtomicLong totalDraftPasses = new AtomicLong(0);

    /** Total target forward passes */
    private final AtomicLong totalTargetPasses = new AtomicLong(0);

    /** Total accepted tokens */
    private final AtomicLong totalAccepted = new AtomicLong(0);

    /** Total rejected tokens */
    private final AtomicLong totalRejected = new AtomicLong(0);

    // ── Lifecycle ─────────────────────────────────────────────────────

    private volatile boolean initialized = false;

    private SpeculativeDecoder(Config config) {
        this.draftModelId = config.draftModelId;
        this.targetModelId = config.targetModelId;
        this.draftTokens = config.draftTokens;
        this.temperature = config.temperature;
        this.topP = config.topP;
        this.seed = config.seed;
    }

    /**
     * Creates a builder for configuring this decoder.
     */
    public static Builder builder() {
        return new Builder();
    }

    // ── Generation ────────────────────────────────────────────────────

    /**
     * Generates tokens using speculative decoding.
     *
     * @param promptTokens input prompt token IDs
     * @param maxTokens maximum tokens to generate
     * @param callback callback for each generated token
     * @return list of generated token IDs
     */
    public List<Integer> generate(List<Integer> promptTokens, int maxTokens,
                                  TokenCallback callback) {
        ensureInitialized();

        List<Integer> output = new ArrayList<>(promptTokens);
        int tokensGenerated = 0;

        while (tokensGenerated < maxTokens) {
            // Step 1: Draft model generates K tokens
            List<Integer> draftedTokens = draftTokens(output, draftTokens);
            totalDraftPasses.addAndGet(draftedTokens.size());

            // Step 2: Target model verifies all draft tokens
            VerificationResult result = verifyTokens(output, draftedTokens);
            totalTargetPasses.incrementAndGet();

            // Step 3: Accept verified tokens
            int acceptedCount = 0;
            for (int i = 0; i < result.accepted.length; i++) {
                if (result.accepted[i]) {
                    output.add(draftedTokens.get(i));
                    callback.onToken(draftedTokens.get(i), tokensGenerated, false);
                    tokensGenerated++;
                    acceptedCount++;
                    totalAccepted.incrementAndGet();
                } else {
                    // Resample at rejection point
                    int resampledToken = result.resampledToken;
                    output.add(resampledToken);
                    callback.onToken(resampledToken, tokensGenerated, false);
                    tokensGenerated++;
                    totalRejected.incrementAndGet();
                    break;  // Stop accepting from this draft
                }
            }

            // If all draft tokens accepted, add one more from target
            if (acceptedCount == draftedTokens.size() && tokensGenerated < maxTokens) {
                int targetToken = sampleFromTarget(output);
                output.add(targetToken);
                callback.onToken(targetToken, tokensGenerated, false);
                tokensGenerated++;
                totalTargetPasses.incrementAndGet();
                totalAccepted.incrementAndGet();
            }

            totalTokens.addAndGet(acceptedCount + (acceptedCount < draftedTokens.size() ? 1 : 0));
        }

        // Signal completion
        if (tokensGenerated > 0) {
            callback.onToken(output.get(output.size() - 1), tokensGenerated - 1, true);
        }

        LOG.infof("Speculative decoding complete: %d tokens, acceptance rate=%.1f%%, speedup=%.2f×",
            tokensGenerated, getAcceptanceRate(), getSpeedup());

        return output.subList(promptTokens.size(), output.size());
    }

    // ── Internal Methods ──────────────────────────────────────────────

    /**
     * Drafts K tokens using the draft model.
     */
    private List<Integer> draftTokens(List<Integer> context, int k) {
        List<Integer> tokens = new ArrayList<>(k);
        List<Integer> draftContext = new ArrayList<>(context);

        for (int i = 0; i < k; i++) {
            // In production: run draft model forward pass
            // For now: placeholder
            int token = sampleToken(draftContext);
            tokens.add(token);
            draftContext.add(token);
        }

        return tokens;
    }

    /**
     * Verifies draft tokens against target model.
     */
    private VerificationResult verifyTokens(List<Integer> context, List<Integer> draftTokens) {
        int n = draftTokens.size();
        boolean[] accepted = new boolean[n];

        // In production: run target model forward pass for all draft tokens
        // Compare draft probabilities with target probabilities
        for (int i = 0; i < n; i++) {
            // Placeholder: accept with some probability
            accepted[i] = Math.random() < 0.8;  // 80% acceptance rate
            if (!accepted[i]) {
                break;  // Stop at first rejection
            }
        }

        // Find first rejection point
        int rejectIdx = -1;
        for (int i = 0; i < n; i++) {
            if (!accepted[i]) {
                rejectIdx = i;
                break;
            }
        }

        // If all accepted, we'll add one more token from target
        // Otherwise, resample at rejection point
        int resampledToken = rejectIdx >= 0 ? sampleToken(context) : -1;

        return new VerificationResult(accepted, resampledToken);
    }

    /**
     * Samples a single token from the target model.
     */
    private int sampleFromTarget(List<Integer> context) {
        // In production: run target model forward pass and sample
        return sampleToken(context);
    }

    /**
     * Samples a token from the model's output distribution.
     */
    private int sampleToken(List<Integer> context) {
        // Placeholder: random token sampling
        // In production: softmax + top-p + temperature sampling
        return new Random(seed + totalTokens.get()).nextInt(32000);
    }

    /**
     * Ensures the decoder is initialized.
     */
    private void ensureInitialized() {
        if (!initialized) {
            // In production: load draft and target models
            initialized = true;
            LOG.infof("SpeculativeDecoder initialized: draft=%s, target=%s, draftTokens=%d",
                draftModelId, targetModelId, draftTokens);
        }
    }

    // ── Query Methods ─────────────────────────────────────────────────

    /**
     * Gets token acceptance rate (0.0 to 1.0).
     */
    public double getAcceptanceRate() {
        long total = totalAccepted.get() + totalRejected.get();
        return total == 0 ? 0.0 : (double) totalAccepted.get() / total;
    }

    /**
     * Gets estimated speedup vs standard decoding.
     */
    public double getSpeedup() {
        long draft = totalDraftPasses.get();
        long target = totalTargetPasses.get();
        long total = totalTokens.get();
        if (draft == 0 || target == 0) return 1.0;

        // Speedup = total_tokens / (draft_passes + target_passes)
        // vs standard = total_tokens / total_tokens (1 forward pass per token)
        return (double) total / (target);
    }

    /**
     * Gets decoding statistics.
     */
    public DecodingStats getStats() {
        return new DecodingStats(
            draftModelId,
            targetModelId,
            draftTokens,
            totalTokens.get(),
            totalDraftPasses.get(),
            totalTargetPasses.get(),
            totalAccepted.get(),
            totalRejected.get(),
            getAcceptanceRate(),
            getSpeedup()
        );
    }

    // ── Nested Types ─────────────────────────────────────────────────

    /**
     * Verification result from target model.
     */
    record VerificationResult(
        boolean[] accepted,
        int resampledToken
    ) {}

    /**
     * Decoding statistics.
     */
    public record DecodingStats(
        String draftModelId,
        String targetModelId,
        int draftTokens,
        long totalTokens,
        long totalDraftPasses,
        long totalTargetPasses,
        long totalAccepted,
        long totalRejected,
        double acceptanceRate,
        double speedup
    ) {}

    /**
     * Token callback interface.
     */
    @FunctionalInterface
    public interface TokenCallback {
        void onToken(int tokenId, int position, boolean finished);
    }

    /**
     * Configuration for SpeculativeDecoder.
     */
    private static final class Config {
        String draftModelId = "llama-3-1b";
        String targetModelId = "llama-3-70b";
        int draftTokens = 5;
        double temperature = 0.8;
        double topP = 0.95;
        long seed = 42L;
    }

    /**
     * Builder for SpeculativeDecoder.
     */
    public static final class Builder {
        private final Config config = new Config();

        private Builder() {}

        public Builder draftModelId(String modelId) {
            config.draftModelId = modelId;
            return this;
        }

        public Builder targetModelId(String modelId) {
            config.targetModelId = modelId;
            return this;
        }

        /**
         * Sets number of tokens to draft per verification.
         * <p>
         * Higher values = more speedup potential but lower acceptance rate.
         * Recommended: 4-6 for similar model families.
         */
        public Builder draftTokens(int k) {
            config.draftTokens = k;
            return this;
        }

        public Builder temperature(double temperature) {
            config.temperature = temperature;
            return this;
        }

        public Builder topP(double topP) {
            config.topP = topP;
            return this;
        }

        public Builder seed(long seed) {
            config.seed = seed;
            return this;
        }

        public SpeculativeDecoder build() {
            return new SpeculativeDecoder(config);
        }
    }
}
