package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;

final class FlashAttentionModelPolicy {
    private final ModelArchitecture architecture;
    private final ModelConfig config;
    private final ModelRuntimeTraits traits;
    private final ModelRuntimeTraits.AttentionRuntimeTraits attention;

    private FlashAttentionModelPolicy(ModelArchitecture architecture, ModelConfig config, ModelRuntimeTraits traits) {
        this.architecture = architecture;
        this.config = config;
        this.traits = traits == null ? ModelRuntimeTraits.fromConfig(config) : traits;
        this.attention = this.traits.attention() == null
                ? ModelRuntimeTraits.AttentionRuntimeTraits.EMPTY
                : this.traits.attention();
    }

    static FlashAttentionModelPolicy resolve(ModelArchitecture architecture, ModelConfig config) {
        ModelRuntimeTraits traits = architecture == null ? null : architecture.runtimeTraits(config);
        if (traits == null) {
            traits = ModelRuntimeTraits.fromConfig(config);
        }
        return new FlashAttentionModelPolicy(architecture, config, traits);
    }

    boolean gemma4Text() {
        return traits.gemma4Text();
    }

    boolean gemma3Text() {
        return traits.gemma3Text();
    }

    boolean perLayerInputPath() {
        return traits.perLayerInputPath();
    }

    boolean useInterleavedRope(boolean legacyGemma4Interleaved, boolean experimentalGemma4SplitHalf) {
        boolean splitHalf = attention.splitHalfRope();
        if (traits.gemma4Text()) {
            if (legacyGemma4Interleaved) {
                splitHalf = false;
            }
            if (experimentalGemma4SplitHalf) {
                splitHalf = true;
            }
        }
        boolean architectureUsesSplitHalf = architecture == null || architecture.usesNeoxRope();
        return !splitHalf && !architectureUsesSplitHalf;
    }

    float resolveAttentionSoftCap() {
        if (attention.attentionSoftCapAppliesToFinalLogitsOnly()) {
            return 0.0f;
        }
        Double configured = config != null ? config.attnLogitSoftcapping() : null;
        if (configured != null && configured > 0) {
            return configured.floatValue();
        }
        return architecture == null ? 0.0f : architecture.defaultAttnSoftCap();
    }

    boolean preferMetalPerHeadRmsNorm() {
        return attention.preferMetalPerHeadRmsNorm();
    }

    boolean allowLegacyMetalAttentionBridge(boolean legacyBridgeEnabled) {
        return !attention.restrictLegacyMetalAttentionBridge() || legacyBridgeEnabled;
    }

    boolean supportsForcedDenseAttention() {
        return attention.supportsForcedDenseAttention();
    }

    boolean preferNativeMetalBf16Linear() {
        return attention.preferNativeMetalBf16Linear();
    }

    boolean disallowBf16ToF16LinearConversion() {
        return attention.disallowBf16ToF16LinearConversion();
    }

    int defaultPagedMetalPrefillMaxTokens(int fallback) {
        int preferred = attention.defaultPagedMetalPrefillMaxTokens();
        return preferred > 0 ? preferred : fallback;
    }

    boolean compactAttentionMatvecCandidate() {
        return attention.compactAttentionMatvecCandidate();
    }

    boolean metalHalfMatvecAutoCandidate() {
        return attention.compactAttentionMatvecCandidate()
                || attention.largeAttentionMatvecCandidate();
    }
}
