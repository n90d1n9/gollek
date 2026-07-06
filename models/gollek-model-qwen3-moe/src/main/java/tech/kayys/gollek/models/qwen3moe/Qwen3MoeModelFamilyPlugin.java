package tech.kayys.gollek.models.qwen3moe;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen3MoeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_moe",
                "Alibaba Qwen3-MoE",
                List.of("qwen3_moe"),
                List.of("Qwen3MoeForCausalLM", "Qwen3MoeModel",
                        "Qwen3MoeForSequenceClassification", "Qwen3MoeForTokenClassification",
                        "Qwen3MoeForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.GGUF, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/qwen3_moe",
                        "tokenizer", "qwen_huggingface_bpe",
                        "direct_safetensor", "pending_qwen3_moe_expert_routing_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("qwen3-moe-hf-bpe"));
    }
}
