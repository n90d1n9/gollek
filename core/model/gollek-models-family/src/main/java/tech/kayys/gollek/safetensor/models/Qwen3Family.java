package tech.kayys.gollek.safetensor.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Qwen-3 architecture (0.6B, 1.7B, 4B, 8B, 14B, 32B) — dense models.
 * Also MoE variants: Qwen3-30B-A3B, Qwen3-235B-A22B.
 *
 * Key new features vs Qwen-2.5:
 * - QK-norm (RMSNorm on Q and K before attention — same as Gemma-3)
 * - Thinking mode support (interleaved <think> ... </think> tokens)
 * - Extended context: 32K native, 128K with YaRN
 *
 * HuggingFace models:
 * Qwen/Qwen3-8B, Qwen/Qwen3-8B-Instruct
 * Qwen/Qwen3-32B, Qwen/Qwen3-32B-Instruct
 * Qwen/Qwen3-30B-A3B (MoE), Qwen/Qwen3-235B-A22B (MoE)
 */
@ApplicationScoped
public class Qwen3Family implements ModelArchitecture {

        @Override
        public String id() {
            return "qwen3";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Qwen3ForCausalLM", "Qwen3MoeForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("qwen3");
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

        // QK-norms (new in Qwen-3)
        public String layerQNorm(int i) {
            return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
        }

        public String layerKNorm(int i) {
            return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
        }
    }
