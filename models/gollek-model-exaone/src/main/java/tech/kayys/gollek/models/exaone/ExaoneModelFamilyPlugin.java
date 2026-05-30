package tech.kayys.gollek.models.exaone;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ExaoneModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "exaone",
                "LG EXAONE / EXAONE MoE",
                List.of("exaone4", "exaone_moe", "exaone-moe"),
                List.of("Exaone4ForCausalLM", "Exaone4Model",
                        "Exaone4ForSequenceClassification", "Exaone4ForTokenClassification",
                        "Exaone4ForQuestionAnswering", "ExaoneMoeForCausalLM",
                        "ExaoneMoeModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.GGUF, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/exaone4,exaone_moe",
                        "tokenizer", "huggingface_bpe",
                        "direct_safetensor", "pending_exaone_attention_and_moe_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("exaone-hf-bpe"));
    }
}
