package tech.kayys.gollek.models.nemotron;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class NemotronModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "nemotron",
                "NVIDIA Nemotron",
                List.of("nemotron"),
                List.of("NemotronForCausalLM", "NemotronModel", "NemotronForSequenceClassification",
                        "NemotronForQuestionAnswering", "NemotronForTokenClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.GGUF, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/nemotron",
                        "tokenizer", "sentencepiece_or_tokenizer_json",
                        "direct_safetensor", "pending_nemotron_relu2_norm_and_rotary_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("nemotron-spm-or-tokenizer-json"));
    }
}
