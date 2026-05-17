package tech.kayys.gollek.train.diffusion.api;

import java.util.Map;
import java.util.Objects;

/**
 * One prompt-side unit for diffusion training.
 */
public record DiffusionPromptSample(
        String prompt,
        String negativePrompt,
        long seed,
        Map<String, Object> metadata) {

    public DiffusionPromptSample {
        prompt = Objects.requireNonNull(prompt, "prompt must not be null");
        negativePrompt = negativePrompt == null ? "" : negativePrompt;
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}
