package tech.kayys.gollek.models.gemma3;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Gemma3ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gemma3",
                "Google Gemma 3 Text",
                List.of("gemma3", "gemma3_text"),
                List.of("Gemma3ForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE,
                        ModelFamilyCapability.GGUF),
                Map.of(
                        "bundle_profile", "optional",
                        "origin", "3rdparty/transformers/src/transformers/models/gemma3",
                        "direct_safetensor", "text_adapter_only",
                        "multimodal_architecture", "Gemma3ForConditionalGeneration",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new Gemma3TextFamily());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("gemma3-spm-bpe"));
    }
}
