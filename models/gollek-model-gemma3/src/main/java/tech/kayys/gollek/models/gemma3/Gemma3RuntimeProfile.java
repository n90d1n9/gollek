package tech.kayys.gollek.models.gemma3;

import tech.kayys.gollek.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelPromptTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

/**
 * Gemma 3 text runtime policy for prompt/BOS and split-RoPE attention behavior.
 */
public final class Gemma3RuntimeProfile {

    private static final boolean GEMMA4_TEXT = false;
    private static final boolean GEMMA3_TEXT = true;
    private static final boolean QWEN_TEXT = false;

    private Gemma3RuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        return ModelRuntimeTraits.builder()
                .gemma3Text()
                .prompt(prompt())
                .attention(ModelAttentionTraitsPolicy.gemma3Text())
                .build();
    }

    public static ModelPromptTraits prompt() {
        return ModelPromptTraits.fromRuntimeFlags(GEMMA4_TEXT, GEMMA3_TEXT, QWEN_TEXT);
    }
}
