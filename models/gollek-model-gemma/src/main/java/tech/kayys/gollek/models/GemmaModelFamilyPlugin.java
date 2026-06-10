package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyUnifiedRuntimeRequirement;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;
import static java.util.Map.entry;

/**
 * Core Gemma model-family descriptor for packaged text inference.
 */
@ApplicationScoped
public class GemmaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gemma",
                "Google Gemma",
                List.of("gemma"),
                List.of("GemmaForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE,
                        ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE, ModelFamilyCapability.GGUF),
                Map.ofEntries(
                        entry("bundle_profile", "core"),
                        entry("origin", "3rdparty/transformers/src/transformers/models/gemma"),
                        entry("direct_safetensor", "ready"),
                        entry("tokenizer", "gemma_sentencepiece_with_audio_vision_processor"),
                        entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new GemmaFamily());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("gemma-spm-bpe"));
    }

    @Override
    public List<ModelFamilyUnifiedRuntimeRequirement> unifiedRuntimeRequirements() {
        // Gemma 4 unified runtime requirements are owned by the dedicated Gemma 4 family.
        return List.of();
    }
}
