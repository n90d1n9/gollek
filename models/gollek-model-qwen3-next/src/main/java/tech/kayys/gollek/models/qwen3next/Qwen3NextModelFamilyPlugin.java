package tech.kayys.gollek.models.qwen3next;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Qwen3NextModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen3_next",
                "Alibaba Qwen3-Next",
                List.of("qwen3_next"),
                List.of("Qwen3NextForCausalLM", "Qwen3NextModel",
                        "Qwen3NextForSequenceClassification", "Qwen3NextForTokenClassification",
                        "Qwen3NextForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.GGUF, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/qwen3_next",
                        "tokenizer", "qwen_huggingface_bpe",
                        "direct_safetensor", "pending_qwen3_next_hybrid_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("qwen3-next-hf-bpe"));
    }
}
