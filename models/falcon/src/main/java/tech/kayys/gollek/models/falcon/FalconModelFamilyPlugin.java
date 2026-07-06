package tech.kayys.gollek.models.falcon;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FalconModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "falcon",
                "TII Falcon",
                List.of("falcon"),
                List.of("FalconForCausalLM", "FalconForSequenceClassification",
                        "FalconForTokenClassification", "FalconForQuestionAnswering"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.GGUF),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/falcon",
                        "direct_safetensor", "pending_non_gated_gelu_ffn",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("falcon-hf-bpe"));
    }
}
