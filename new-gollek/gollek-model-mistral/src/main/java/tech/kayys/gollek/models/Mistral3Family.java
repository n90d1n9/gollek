package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Mistral-3 / Mistral-Small architecture (22B, 24B).
 *
 * Key differences from Mistral-7B-v0.3:
 * - Larger: 22B or 24B parameters
 * - Full context (no sliding window in Mistral-Small-3.1)
 * - v3 Tekken tokenizer (131K vocabulary)
 * - Flash Attention 2 trained
 *
 * HuggingFace models:
 * mistralai/Mistral-Small-3.1-22B-Instruct-2503
 * mistralai/Mistral-Small-24B-Instruct-2501
 */
@ApplicationScoped
public class Mistral3Family implements ModelArchitecture {

        @Override
        public String id() {
            return "mistral3";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("MistralForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("mistral3", "mistral");
        }

        // Same weight names as Mistral-7B
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
            return "lm_head.weight";
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
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
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
    }