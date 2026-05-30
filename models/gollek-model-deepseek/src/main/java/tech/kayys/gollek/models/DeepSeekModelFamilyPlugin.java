package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelArchitecture;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DeepSeekModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deepseek",
                "DeepSeek-R1",
                List.of("deepseek_r1"),
                List.of("DeepseekV3ForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.GGUF),
                Map.of(
                        "bundle_profile", "experimental",
                        "origin", "deepseek_r1_compatibility_namespace",
                        "direct_safetensor", "pending_r1_distill_and_v3_flagship_runtime_policy",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new DeepSeekR1Family());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("deepseek-hf-bpe"));
    }
}
