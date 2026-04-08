package tech.kayys.gollek.safetensor.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Google Gemma-2 architecture (2B, 9B, 27B).
 *
 * Key differences from Gemma-1:
 * - Alternating local (sliding window=4096) and global attention layers
 * - post_feedforward_layernorm (extra norm AFTER the FFN output)
 * - Logit soft-capping: clamp logits to tanh(logits / 30) * 30
 * - pre_feedforward_layernorm (extra norm BEFORE the FFN input)
 * - Attention logit soft-capping per head
 * - GeGLU activation (GELU gating, not SiLU)
 *
 * Weight naming:
 * model.layers.{i}.self_attn.q_proj.weight
 * model.layers.{i}.pre_feedforward_layernorm.weight ← UNIQUE to Gemma-2
 * model.layers.{i}.post_feedforward_layernorm.weight ← UNIQUE to Gemma-2
 *
 * HuggingFace models:
 * google/gemma-2-2b, google/gemma-2-2b-it
 * google/gemma-2-9b, google/gemma-2-9b-it
 * google/gemma-2-27b, google/gemma-2-27b-it
 */
@ApplicationScoped
public class Gemma2Family implements ModelArchitecture {

        @Override
        public String id() {
            return "gemma2";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Gemma2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("gemma2");
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
        } // tied

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

        // Gemma-2 has BOTH pre-attention AND pre-feedforward norms (4 norms per layer)
        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }

        /** Pre-feedforward norm (applied before gate/up projections). */
        public String layerPreFfnNormWeight(int i) {
            return "model.layers.%d.pre_feedforward_layernorm.weight".formatted(i);
        }

        /** Post-feedforward norm (applied after down projection). */
        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
        }

        /** Post-attention norm (applied after o_proj residual). */
        public String layerPostAttnNormWeight(int i) {
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