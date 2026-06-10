package tech.kayys.gollek.models;

import tech.kayys.gollek.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelPromptTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.util.Locale;

/**
 * Legacy Gemma runtime policy, including the Gemma4-compatible direct path.
 */
public final class GemmaRuntimeProfile {

    private static final boolean GEMMA3_TEXT = false;
    private static final boolean QWEN_TEXT = false;
    private static final boolean GEMMA_FAMILY = true;

    private GemmaRuntimeProfile() {
    }

    public static ModelRuntimeTraits text(ModelConfig config) {
        boolean gemma4Text = isGemma4Text(config);
        boolean perLayerInputs = perLayerInputPath(config);
        ModelPromptTraits prompt = prompt(config);
        return ModelRuntimeTraits.builder()
                .gemma4Text(gemma4Text)
                .perLayerInputPath(perLayerInputs)
                .prompt(prompt)
                .attention(gemma4Text
                        ? ModelAttentionTraitsPolicy.gemma4Text()
                        : ModelAttentionTraitsPolicy.generic(config, perLayerInputs))
                .build();
    }

    public static ModelPromptTraits prompt(ModelConfig config) {
        return ModelPromptTraits.fromFlags(isGemma4Text(config), GEMMA3_TEXT, GEMMA_FAMILY, QWEN_TEXT);
    }

    static boolean isGemma4Text(ModelConfig config) {
        return normalizedModelType(config).startsWith("gemma4");
    }

    static boolean perLayerInputPath(ModelConfig config) {
        return config != null
                && (config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0);
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }
}
