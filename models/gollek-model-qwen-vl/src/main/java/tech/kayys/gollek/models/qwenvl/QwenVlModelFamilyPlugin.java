package tech.kayys.gollek.models.qwenvl;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class QwenVlModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "qwen_vl",
                "Legacy Qwen-VL",
                List.of("qwen_vl"),
                List.of("QwenVLForConditionalGeneration", "QwenVLModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "legacy/qwen_vl_compatibility_namespace",
                        "tokenizer", "qwen_hf_bpe_legacy_vl",
                        "direct_safetensor", "pending_legacy_qwen_vl_runtime_contract",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("qwen-vl-hf-bpe"));
    }
}
