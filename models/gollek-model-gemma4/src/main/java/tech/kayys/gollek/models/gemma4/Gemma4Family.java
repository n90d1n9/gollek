package tech.kayys.gollek.models.gemma4;

import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits.PromptBosPolicy;

import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Google Gemma 4 text adapter. Multimodal wrappers stay guarded by runtime checks.
 */
public class Gemma4Family implements ModelArchitecture {
    @Override
    public String id() {
        return "gemma4";
    }

    @Override
    public FFNActivationType activationType() {
        return FFNActivationType.GELU;
    }

    @Override
    public boolean usesNeoxRope() {
        return true;
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma4", "gemma4_text");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    public String embedTokensPerLayerWeight() {
        return "model.embed_tokens_per_layer.weight";
    }

    @Override
    public String perLayerModelProjectionWeight() {
        return "model.per_layer_model_projection.weight";
    }

    @Override
    public String perLayerProjectionNormWeight() {
        return "model.per_layer_projection_norm.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public String layerQueryWeight(int i) {
        return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
    }

    @Override
    public String layerKeyWeight(int i) {
        return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
    }

    @Override
    public String layerOutputWeight(int i) {
        return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
    }

    @Override
    public String layerAttentionNormWeight(int i) {
        return "model.layers.%d.input_layernorm.weight".formatted(i);
    }

    @Override
    public String layerQueryNormWeight(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    @Override
    public String layerKeyNormWeight(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    @Override
    public String layerPostAttnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPerLayerInputGateWeight(int i) {
        return "model.layers.%d.per_layer_input_gate.weight".formatted(i);
    }

    @Override
    public String layerPerLayerProjectionWeight(int i) {
        return "model.layers.%d.per_layer_projection.weight".formatted(i);
    }

    @Override
    public String layerPostPerLayerInputNormWeight(int i) {
        return "model.layers.%d.post_per_layer_input_norm.weight".formatted(i);
    }

    @Override
    public String layerScalarWeight(int i) {
        return "model.layers.%d.layer_scalar".formatted(i);
    }

    @Override
    public String layerFfnGateWeight(int i) {
        return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.mlp.up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers.%d.mlp.down_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public String layerPostFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
    }

    @Override
    public float embeddingScaleFactor(int hiddenDim) {
        return (float) Math.sqrt(hiddenDim);
    }

    @Override
    public boolean addOneToRmsNormWeight() {
        return true;
    }

    @Override
    public ModelRuntimeTraits runtimeTraits(ModelConfig config) {
        String modelType = config == null || config.modelType() == null
                ? ""
                : config.modelType().toLowerCase(Locale.ROOT);
        boolean gemma4Text = modelType.startsWith("gemma4");
        boolean perLayerInputs = config != null
                && (config.hiddenSizePerLayerInput() > 0 || config.vocabSizePerLayerInput() > 0);
        return new ModelRuntimeTraits(
                gemma4Text,
                false,
                false,
                perLayerInputs,
                gemma4Text ? PromptBosPolicy.NEVER : PromptBosPolicy.GEMMA_TURN_AWARE,
                gemma4Text ? ModelRuntimeTraits.GEMMA4_CONTROL_TOKEN_TEXTS : Set.of(),
                gemma4Text,
                gemma4Text,
                gemma4Text
                        ? ModelRuntimeTraits.AttentionRuntimeTraits.gemma4Text()
                        : ModelRuntimeTraits.AttentionRuntimeTraits.generic(config, perLayerInputs));
    }

    @Override
    public float defaultAttnSoftCap() {
        return 0.0f;
    }

    @Override
    public float defaultFinalSoftCap() {
        return 0.0f;
    }
}
