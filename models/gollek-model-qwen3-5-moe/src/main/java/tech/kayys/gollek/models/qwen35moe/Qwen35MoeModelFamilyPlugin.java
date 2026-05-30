package tech.kayys.gollek.models.qwen35moe;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen35MoeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_5_moe",
                "Alibaba Qwen3.5-MoE",
                List.of("qwen3_5_moe", "qwen3_5_moe_text", "qwen3_5_moe_vision"),
                List.of("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration",
                        "Qwen3_5MoeModel", "Qwen3_5MoeTextModel", "Qwen3_5MoeVisionModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/qwen3_5_moe",
                        "tokenizer", "qwen3_5_tokenizer_with_moe_vision_processor",
                        "direct_safetensor", "pending_qwen3_5_moe_vision_expert_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "qwen3-5-moe-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_qwen3_5",
                        "status", "metadata_only")));
    }
}
