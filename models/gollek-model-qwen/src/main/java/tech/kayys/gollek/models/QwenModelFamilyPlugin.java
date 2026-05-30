package tech.kayys.gollek.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class QwenModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen",
                "Alibaba Qwen",
                List.of("qwen2", "qwen2.5", "qwen3"),
                List.of("Qwen2ForCausalLM", "Qwen3ForCausalLM"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE,
                        ModelFamilyCapability.GGUF),
                Map.of(
                        "bundle_profile", "core",
                        "origin", "3rdparty/transformers/src/transformers/models/qwen2,qwen3",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new Qwen2Family(), new Qwen25Family(), new Qwen3Family());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("qwen-hf-bpe"));
    }
}
