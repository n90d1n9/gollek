package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

    /**
     * DeepSeek-R1 architecture (7B, 8B, 14B, 32B, 70B distillations + 671B MoE).
     *
     * DeepSeek-R1 distillations use Qwen-2.5 or LLaMA-3 as base models.
     * The 671B flagship is a MoE model using DeepSeek-V3 architecture.
     * R1 adds chain-of-thought reasoning via extended <think> token generation.
     *
     * HuggingFace models:
     * deepseek-ai/DeepSeek-R1-Distill-Qwen-7B → qwen2 arch
     * deepseek-ai/DeepSeek-R1-Distill-LLama-8B → llama arch
     * deepseek-ai/DeepSeek-R1-Distill-Qwen-32B → qwen2 arch
     * deepseek-ai/DeepSeek-R1 (671B) → deepseek_v3 arch
     */
    @ApplicationScoped
    public  class DeepSeekR1Family implements ModelArchitecture {

        @Override
        public String id() {
            return "deepseek_r1";
        }

        @Override
        public List<String> supportedArchClassNames() {
            // R1 distillations reuse existing arch classes
            return List.of("DeepseekV3ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("deepseek_r1");
        }

        // R1 671B uses same weights as DeepSeek-V3
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
            return "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.layers.%d.self_attn.kv_b_proj.weight".formatted(i);
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