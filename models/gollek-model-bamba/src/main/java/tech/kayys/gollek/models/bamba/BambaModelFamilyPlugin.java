package tech.kayys.gollek.models.bamba;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BambaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bamba",
                "AI21 Bamba",
                List.of("bamba"),
                List.of("BambaForCausalLM", "BambaModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/bamba",
                        "tokenizer", "llama_compatible_tokenizer",
                        "direct_safetensor", "pending_bamba_hybrid_attention_mamba_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("bamba-spm-bpe"));
    }
}
