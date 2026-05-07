package tech.kayys.gollek.gguf.loader.sampler;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class Sampler {
    private final float temperature;
    private final int topK;
    private final float topP;
    private final float repeatPenalty;
    
    public Sampler(float temperature, int topK, float topP, float repeatPenalty) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.repeatPenalty = repeatPenalty;
    }
    
    public int sample(float[] logits, List<Integer> previousTokens) {
        float[] probs = logits.clone();
        
        // Apply repetition penalty
        for (int token : previousTokens) {
            if (probs[token] < 0) probs[token] *= repeatPenalty;
            else probs[token] /= repeatPenalty;
        }
        
        // Apply temperature
        if (temperature > 0) {
            for (int i = 0; i < probs.length; i++) probs[i] /= temperature;
        }
        
        // Top-K
        if (topK > 0 && topK < probs.length) {
            float[] sorted = probs.clone();
            java.util.Arrays.sort(sorted);
            float threshold = sorted[probs.length - topK];
            for (int i = 0; i < probs.length; i++) {
                if (probs[i] < threshold) probs[i] = Float.NEGATIVE_INFINITY;
            }
        }
        
        // Softmax
        float max = probs[0];
        for (float p : probs) if (p > max) max = p;
        float sum = 0;
        for (int i = 0; i < probs.length; i++) {
            if (!Float.isInfinite(probs[i])) {
                probs[i] = (float) Math.exp(probs[i] - max);
                sum += probs[i];
            } else {
                probs[i] = 0;
            }
        }
        float invSum = 1f / sum;
        for (int i = 0; i < probs.length; i++) probs[i] *= invSum;
        
        // Sample
        double r = ThreadLocalRandom.current().nextDouble();
        double cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r < cum) return i;
        }
        return 0;
    }
}
