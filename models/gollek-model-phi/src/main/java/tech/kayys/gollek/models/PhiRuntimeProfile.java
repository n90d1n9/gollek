package tech.kayys.gollek.models;

import tech.kayys.gollek.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelPromptTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

/**
 * Phi-specific runtime policy for prompt and packed-QKV attention behavior.
 */
public final class PhiRuntimeProfile {

    private static final boolean GEMMA4_TEXT = false;
    private static final boolean GEMMA3_TEXT = false;
    private static final boolean QWEN_TEXT = false;

    private PhiRuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        return ModelRuntimeTraits.builder()
                .prompt(prompt())
                .attention(ModelAttentionTraitsPolicy.phiText(config))
                .build();
    }

    public static ModelPromptTraits prompt() {
        return ModelPromptTraits.fromRuntimeFlags(GEMMA4_TEXT, GEMMA3_TEXT, QWEN_TEXT);
    }
}
