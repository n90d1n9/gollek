package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Google Gemma-3 text architecture (1B, 4B, 12B, 27B).
 *
 * Key differences from Gemma-2:
 * - 5:1 ratio of local (sliding window) to global attention layers
 * - Longer context: 128K tokens (global layers), 1024 tokens (local layers)
 * - QK-norm (separate RMSNorm on Q and K before attention)
 * - 256K vocabulary (tiktoken-based, same as Gemma-2)
 * - GeGLU activation
 *
 * HuggingFace models:
 * google/gemma-3-1b-it, google/gemma-3-4b-it
 * google/gemma-3-12b-it, google/gemma-3-27b-it
 */
@ApplicationScoped
public class Gemma3TextFamily implements ModelArchitecture {

    @Override
    public String id() {
        return "gemma3_text";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("Gemma3ForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("gemma3");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.norm.weight";
    }

    @Override
    public String lmHeadWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public boolean hasTiedEmbeddings() {
        return true;
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
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
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

    // QK-norms (Gemma-3 applies RMSNorm to Q and K before scaled dot-product)
    public String layerQNorm(int i) {
        return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
    }

    public String layerKNorm(int i) {
        return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
    }

    public String layerPostAttnNorm(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }
}
