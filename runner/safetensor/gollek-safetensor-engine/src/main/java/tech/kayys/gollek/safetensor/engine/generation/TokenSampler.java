/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;

import java.util.Random;

/**
 * Token sampler for autoregressive LLM generation.
 *
 * <p>
 * Converts raw logits into a sampled next-token ID using the strategy
 * specified in {@link GenerationConfig}:
 * <ul>
 * <li><b>Greedy</b> — returns the argmax when requested by strategy,
 * {@code temperature < 1e-4}, or {@code topK == 1}.</li>
 * <li><b>Temperature sampling</b> — scales logits by {@code 1/temperature}
 * before softmax, then samples from the resulting distribution.</li>
 * </ul>
 *
 * <p>
 * Top-k and top-p (nucleus) filtering can be layered on top by pre-processing
 * the logits before calling {@link #sample}.
 */
@ApplicationScoped
public class TokenSampler {

    private static final String VALIDATE_LOGITS_PROPERTY = "gollek.safetensor.validate_logits";

    private final Random random = new Random();
    private final ThreadLocal<SamplingWorkspace> workspaces = ThreadLocal.withInitial(SamplingWorkspace::new);

    /**
     * Samples the next token from the given logits using the default internal RNG.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration (temperature, strategy, etc.)
     * @param freq   per-token frequency counts used for repetition penalties
     *               (currently reserved; pass an empty array if unused)
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or
     *         empty
     */
    public int sample(float[] logits, GenerationConfig config, int[] freq) {
        return sample(logits, config, null, freq);
    }

    public int sample(float[] logits, GenerationConfig config, tech.kayys.gollek.spi.model.ModelConfig modelConfig, int[] freq) {
        return sample(logits, config, modelConfig, freq, random);
    }

    /**
     * Samples the next token using a caller-supplied {@link Random} instance.
     *
     * <p>
     * Useful for reproducible generation — pass a seeded {@code Random} to
     * get deterministic output.
     *
     * @param logits raw logit scores for each vocabulary token
     * @param config generation configuration
     * @param freq   per-token frequency counts (reserved)
     * @param rng    random number generator to use for sampling
     * @return the sampled token ID, or {@code -1} if {@code logits} is null or
     *         empty
     */
    public int sample(float[] logits, GenerationConfig config, tech.kayys.gollek.spi.model.ModelConfig modelConfig, int[] freq, Random rng) {
        if (logits == null || logits.length == 0) {
            System.err.println("ERROR: Empty or null logits array");
            return -1;
        }

        if (Boolean.getBoolean(VALIDATE_LOGITS_PROPERTY) && !validateLogits(logits)) {
            return -1;
        }

        // Extract values
        float temp = (float) config.temperature();
        int topK = config.topK();
        float topP = (float) config.topP();
        float minP = (float) config.minP();

        // Apply repetition and frequency penalties (applies to both greedy and sampled
        // modes)
        float repPenalty = config.repetitionPenalty();
        float freqPenalty = config.frequencyPenalty();
        if (freq != null && (repPenalty > 1.0f || freqPenalty > 0.0f)) {
            for (int i = 0; i < logits.length && i < freq.length; i++) {
                if (freq[i] > 0) {
                    if (repPenalty > 1.0f) {
                        float effectiveRepPenalty = freq[i] > 1
                                ? (float) Math.pow(repPenalty, freq[i])
                                : repPenalty;
                        if (logits[i] > 0) {
                            logits[i] /= effectiveRepPenalty;
                        } else {
                            logits[i] *= effectiveRepPenalty;
                        }
                    }
                    if (freqPenalty > 0.0f) {
                        logits[i] -= freqPenalty * freq[i];
                    }
                }
            }
        }

        // For deterministic decoding, aggressively suppress prompt-echo tokens once
        // they have already been generated on top of their prompt occurrence. This
        // helps short local QA prompts avoid falling into `jakarta jakarta ...` style
        // loops while still letting the model use the prompt vocabulary for the first
        // step.
        boolean greedy = config.requestsGreedyDecoding();
        if (greedy && freq != null && repPenalty > 1.0f) {
            for (int i = 0; i < logits.length && i < freq.length; i++) {
                if (freq[i] > 1) {
                    logits[i] = Float.NEGATIVE_INFINITY;
                }
            }
        }

        // Greedy sampling when the effective sampling policy is deterministic.
        if (greedy) {
            int best = -1;
            float bestVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < logits.length; i++) {
                float value = logits[i];
                if (!Float.isNaN(value) && value > bestVal) {
                    bestVal = value;
                    best = i;
                }
            }
            return best >= 0 ? best : -1;
        }

        if (temp <= 0)
            temp = 1.0f;

        SamplingWorkspace workspace = workspaces.get();
        int[] indices = workspace.indices(logits.length);
        for (int i = 0; i < logits.length; i++) indices[i] = i;

        // O(N) QuickSelect to find the top-K elements without sorting the entire vocabulary
        int limit = (topK > 0 && topK < logits.length) ? topK : logits.length;
        if (limit < logits.length) {
            quickSelect(logits, indices, 0, logits.length - 1, limit);
        }
        
        // Sort only the top-K elements for sampling
        iterativeQuickSort(logits, indices, 0, limit - 1, limit);


        // Max logit is now at indices[0]
        float maxLogit = logits[indices[0]];

        // Softmax with temperature
        double[] probs = workspace.probs(limit);
        double sum = 0;

        // Apply softcapping only to the top-K candidates (huge O(N) -> O(K) optimization)
        Double cap = modelConfig == null ? null : modelConfig.finalLogitSoftcapping();
        float softCap = (cap != null && cap > 0) ? cap.floatValue() : 0.0f;
        float normalizationMaxLogit = softCap > 0
                ? (float) (softCap * Math.tanh(maxLogit / softCap))
                : maxLogit;

        // Apply Min-P limit
        double minProbLimit = 0.0;
        if (minP > 0.0f) {
            minProbLimit = minP * Math.exp((maxLogit - maxLogit) / temp); // which is just minP
        }

        int actualElements = 0;
        for (int i = 0; i < limit; i++) {
            int idx = indices[i];
            float logit = logits[idx];
            if (softCap > 0) {
                logit = (float) (softCap * Math.tanh(logit / softCap));
            }
            double p = Math.exp((logit - normalizationMaxLogit) / temp);

            if (minP > 0.0f && p < minProbLimit && actualElements > 0) {
                // If it falls below threshold and we already have at least 1 token, stop.
                break;
            }

            probs[i] = p;
            sum += p;
            actualElements++;
        }

        // Top-P filter
        if (topP > 0.0f && topP < 1.0f) {
            double cumulative = 0.0;
            for (int i = 0; i < actualElements; i++) {
                double pNorm = probs[i] / sum;
                cumulative += pNorm;
                if (cumulative > topP && i > 0) {
                    actualElements = i + 1;
                    break;
                }
            }
        }

        // Recompute sum over filtered elements for exact normalization
        double filteredSum = 0;
        for (int i = 0; i < actualElements; i++) {
            filteredSum += probs[i];
        }

        double r = rng.nextDouble() * filteredSum;
        double cumulative = 0;
        for (int i = 0; i < actualElements; i++) {
            cumulative += probs[i];
            if (cumulative >= r) {
                return indices[i];
            }
        }

        return indices[0];
    }

    private static boolean validateLogits(float[] logits) {
        int nanCount = 0;
        int infCount = 0;
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (Float.isNaN(logit)) {
                nanCount++;
            }
            if (Float.isInfinite(logit)) {
                infCount++;
            }
            if (logit < minVal) {
                minVal = logit;
            }
            if (logit > maxVal) {
                maxVal = logit;
            }
        }

        if (nanCount > logits.length * 0.1 || infCount > logits.length * 0.1) {
            System.err.println("WARNING: Corrupted logits detected!");
            System.err.println("  NaN count: " + nanCount + "/" + logits.length);
            System.err.println("  Inf count: " + infCount + "/" + logits.length);
            System.err.println("  Range: [" + minVal + ", " + maxVal + "]");
            System.err.println("  First 10 logits: " + java.util.Arrays.toString(
                    java.util.Arrays.copyOf(logits, Math.min(10, logits.length))));
            return false;
        }
        return true;
    }

    private static final class SamplingWorkspace {
        private int[] indices;
        private double[] probs;

        int[] indices(int size) {
            if (indices == null || indices.length < size) {
                indices = new int[size];
            }
            return indices;
        }

        double[] probs(int size) {
            if (probs == null || probs.length < size) {
                probs = new double[size];
            }
            return probs;
        }
    }

    private void iterativeQuickSort(float[] values, int[] indices, int low, int high, int limit) {
        int[] stack = new int[high - low + 1];
        int top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            high = stack[top--];
            low = stack[top--];

            int p = partition(values, indices, low, high);

            // If we have enough sorted elements for our limit, we can skip deeper partitions
            if (low < p - 1) {
                stack[++top] = low;
                stack[++top] = p - 1;
            }

            // Only proceed with the right partition if the left side didn't already satisfy the limit
            if (p + 1 < high && p < limit) {
                stack[++top] = p + 1;
                stack[++top] = high;
            }
        }
    }

    private void quickSelect(float[] values, int[] indices, int low, int high, int k) {
        while (low < high) {
            int p = partition(values, indices, low, high);
            if (p == k) return;
            if (k < p) {
                high = p - 1;
            } else {
                low = p + 1;
            }
        }
    }

    private int partition(float[] values, int[] indices, int low, int high) {
        float pivot = values[indices[high]];
        int i = (low - 1);
        for (int j = low; j <= high - 1; j++) {
            if (values[indices[j]] > pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        return i + 1;
    }
}
