package tech.kayys.gollek.models;

import tech.kayys.gollek.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelPromptTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

/**
 * Qwen-specific runtime policy for prompt and attention behavior.
 *
 * <p>The architecture adapters own this family profile so new Qwen variants can
 * evolve here instead of scattering runtime defaults across each adapter.</p>
 */
public final class QwenRuntimeProfile {

    private static final boolean GEMMA4_TEXT = false;
    private static final boolean GEMMA3_TEXT = false;
    private static final boolean QWEN_TEXT = true;

    private QwenRuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        return ModelRuntimeTraits.builder()
                .qwenText()
                .prompt(prompt())
                .attention(ModelAttentionTraitsPolicy.qwenText(config))
                .build();
    }

    public static ModelPromptTraits prompt() {
        return ModelPromptTraits.fromRuntimeFlags(GEMMA4_TEXT, GEMMA3_TEXT, QWEN_TEXT);
    }
}
