package tech.kayys.gollek.safetensor.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Gemma / Gemma2 family (Google DeepMind).
 * Almost identical to LLaMA but uses GeGLU and different norm names.
 */
@ApplicationScoped
public class GemmaFamily implements ModelArchitecture {
    @Override
    public String id() {
        return "gemma";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma4ForConditionalGeneration");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma", "gemma2", "gemma4");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    public String embedTokensPerLayerWeight() {
        return "model.embed_tokens_per_layer.weight";
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

    // QK-norms (Gemma-3/4 apply RMSNorm to Q and K before attention)
    public String layerQNorm(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    public String layerKNorm(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    // Post-attention norm (Gemma-2/3/4)
    public String layerPostAttnNorm(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    // Pre-FFN norm (Gemma-2/4)
    public String layerPreFfnNormWeight(int i) {
        return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
    }

    // Per-layer input gating (Gemma-4)
    public String layerPerLayerInputGateWeight(int i) {
        return "model.layers.%d.per_layer_input_gate.weight".formatted(i);
    }

    public String layerPerLayerProjectionWeight(int i) {
        return "model.layers.%d.per_layer_projection.weight".formatted(i);
    }

    public String layerPostPerLayerInputNormWeight(int i) {
        return "model.layers.%d.post_per_layer_input_norm.weight".formatted(i);
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
}
