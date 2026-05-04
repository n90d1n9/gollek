package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Phi-3 family (Microsoft).
 * Uses fused QKV and a different FFN structure.
 */
@ApplicationScoped
public class PhiFamily implements ModelArchitecture {
    @Override
    public String id() {
        return "phi";
    }

    @Override
    public List<String> supportedArchClassNames() {
        return List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM");
    }

    @Override
    public List<String> supportedModelTypes() {
        return List.of("phi", "phi3");
    }

    @Override
    public String embedTokensWeight() {
        return "model.embed_tokens.weight";
    }

    @Override
    public String finalNormWeight() {
        return "model.final_layernorm.weight";
    }

    @Override
    public String finalNormBias() {
        return "model.final_layernorm.bias";
    }

    @Override
    public boolean hasFusedQKV() {
        return true;
    }

    @Override
    public String layerFusedQKVWeight(int i) {
        return "model.layers.%d.self_attn.qkv_proj.weight".formatted(i);
    }

    @Override
    public String layerQueryWeight(int i) {
        return layerFusedQKVWeight(i);
    } // not used separately

    @Override
    public String layerKeyWeight(int i) {
        return layerFusedQKVWeight(i);
    }

    @Override
    public String layerValueWeight(int i) {
        return layerFusedQKVWeight(i);
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
    public String layerFfnGateWeight(int i) {
        return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnUpWeight(int i) {
        return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnDownWeight(int i) {
        return "model.layers.%d.mlp.down_proj.weight".formatted(i);
    }

    @Override
    public String layerFfnNormWeight(int i) {
        return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
    }

    @Override
    public boolean usesRmsNorm() {
        return false;
    }
}