package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Optimized kernel for sampling the next token from logit distribution.
 * Performance optimized for large vocabularies (e.g. Qwen2 with 151k tokens).
 */
public final class SamplingKernel {

    public static int sample(
        MemorySegment logits, 
        int vocabSize, 
        float temp, 
        int topK, 
        float topP,
        float repetitionPenalty,
        List<Integer> recentTokens
    ) {
        // Use a local float buffer for probabilities to avoid repeated MemorySegment access
        float[] p = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            p[i] = logits.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
        }

        // 0. Apply repetition penalty
        if (repetitionPenalty != 1.0f && recentTokens != null && !recentTokens.isEmpty()) {
            for (int tokenId : recentTokens) {
                if (tokenId >= 0 && tokenId < vocabSize) {
                    if (p[tokenId] > 0) {
                        p[tokenId] /= repetitionPenalty;
                    } else {
                        p[tokenId] *= repetitionPenalty;
                    }
                }
            }
        }

        // Greedy decoding (fast path)
        if (temp <= 0.0f) {
            int maxIdx = 0;
            float maxV = p[0];
            for (int i = 1; i < vocabSize; i++) {
                if (p[i] > maxV) {
                    maxV = p[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        // 1. Temperature scaling
        if (temp != 1.0f) {
            float invTemp = 1.0f / temp;
            for (int i = 0; i < vocabSize; i++) p[i] *= invTemp;
        }

        // 2. Softmax (Numerical stability: subtract max)
        float maxVal = p[0];
        for (int i = 1; i < vocabSize; i++) if (p[i] > maxVal) maxVal = p[i];
        
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; i++) {
            p[i] = (float) Math.exp(p[i] - maxVal);
            sum += p[i];
        }
        float invSum = 1.0f / sum;
        for (int i = 0; i < vocabSize; i++) p[i] *= invSum;

        // 3. Optimized Top-K Selection
        // Instead of sorting 151k elements, we find the top K and only work with them.
        int effectiveK = (topK > 0 && topK < vocabSize) ? topK : vocabSize;
        int[] topIndices = new int[effectiveK];
        float[] topProbs = new float[effectiveK];
        
        // Simple selection: if effectiveK is small (e.g. 40), we can just do a linear scan or use a local heap structure.
        // For simplicity and speed with small K, we'll keep a sorted list of the best seen so far.
        // If effectiveK is large (near vocabSize), this becomes O(V*K), which is bad. 
        // But in LLM inference, topK is almost always 40, 50, or 100.
        if (effectiveK < 500) {
            int count = 0;
            for (int i = 0; i < vocabSize; i++) {
                float prob = p[i];
                if (count < effectiveK) {
                    // Insertion sort into the active window
                    int j = count - 1;
                    while (j >= 0 && topProbs[j] < prob) {
                        topProbs[j+1] = topProbs[j];
                        topIndices[j+1] = topIndices[j];
                        j--;
                    }
                    topProbs[j+1] = prob;
                    topIndices[j+1] = i;
                    count++;
                } else if (prob > topProbs[effectiveK - 1]) {
                    // Replace the worst and bubble up
                    int j = effectiveK - 2;
                    while (j >= 0 && topProbs[j] < prob) {
                        topProbs[j+1] = topProbs[j];
                        topIndices[j+1] = topIndices[j];
                        j--;
                    }
                    topProbs[j+1] = prob;
                    topIndices[j+1] = i;
                }
            }
        } else {
            // Fallback for large K: collect all and sort (rare in practice)
            // But still using primitive arrays to avoid GC pressure
            int[] allIndices = new int[vocabSize];
            for (int i = 0; i < vocabSize; i++) allIndices[i] = i;
            
            // Partial sort or full sort (using a simple selection algorithm if needed)
            // For now, full sort if K is large, but we'll optimize the TokenProb closure.
            quickSort(p, allIndices, 0, vocabSize - 1);
            for (int i = 0; i < effectiveK; i++) {
                topIndices[i] = allIndices[i];
                topProbs[i] = p[allIndices[i]];
            }
        }

        // 4. Top-P (Nucleus)
        float currentSum = 0.0f;
        int lastIdx = effectiveK;
        if (topP > 0 && topP < 1.0f) {
            // Re-normalize top K first
            float topSum = 0.0f;
            for (int i = 0; i < effectiveK; i++) topSum += topProbs[i];
            float invTopSum = 1.0f / topSum;
            
            float cumulative = 0.0f;
            for (int i = 0; i < effectiveK; i++) {
                float normalizedProb = topProbs[i] * invTopSum;
                cumulative += normalizedProb;
                if (cumulative > topP) {
                    lastIdx = i + 1;
                    break;
                }
            }
        }

        // 5. Sample from the filtered set
        float finalSum = 0.0f;
        for (int i = 0; i < lastIdx; i++) finalSum += topProbs[i];
        float invFinalSum = 1.0f / finalSum;

        float r = ThreadLocalRandom.current().nextFloat();
        float acc = 0.0f;
        for (int i = 0; i < lastIdx; i++) {
            acc += topProbs[i] * invFinalSum;
            if (r <= acc) return topIndices[i];
        }

        return topIndices[0];
    }

    private static void quickSort(float[] probs, int[] indices, int low, int high) {
        if (low < high) {
            int pivotIdx = partition(probs, indices, low, high);
            quickSort(probs, indices, low, pivotIdx - 1);
            quickSort(probs, indices, pivotIdx + 1, high);
        }
    }

    private static int partition(float[] probs, int[] indices, int low, int high) {
        float pivot = probs[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (probs[indices[j]] > pivot) { // Descending
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

    public static int sample(
        MemorySegment logits,
        int vocabSize,
        float temp,
        int topK,
        float topP
    ) {
        return sample(logits, vocabSize, temp, topK, topP, 1.0f, null);
    }
}
