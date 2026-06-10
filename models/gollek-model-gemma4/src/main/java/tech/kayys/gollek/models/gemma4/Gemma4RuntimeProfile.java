package tech.kayys.gollek.models.gemma4;

import tech.kayys.gollek.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelPromptTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

import java.util.Locale;

/**
 * Gemma 4 text runtime policy for prompt/control-token and attention behavior.
 */
public final class Gemma4RuntimeProfile {

    private static final boolean GEMMA3_TEXT = false;
    private static final boolean QWEN_TEXT = false;
    private static final boolean GEMMA_FAMILY = true;

    private Gemma4RuntimeProfile() {
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
        // Gemma 4 unified 12B may expose vocab_size_per_layer_input without
        // shipping PLE tensors. The hidden size is the executable PLE signal.
        return config != null && config.hiddenSizePerLayerInput() > 0;
    }

    private static String normalizedModelType(ModelConfig config) {
        return config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
    }
}
