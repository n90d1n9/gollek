package tech.kayys.gollek.models.vipllava;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VipLlavaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vipllava",
                "Vip-LLaVA",
                List.of("vipllava", "vip_llava", "vip-llava"),
                List.of("VipLlavaForConditionalGeneration", "VipLlavaModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/vipllava",
                        "tokenizer", "delegated_llama_or_clip_tokenizer",
                        "image_processor", "vipllava_image_processor",
                        "direct_safetensor", "pending_vipllava_visual_prompt_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("vipllava-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("vipllava-spm-bpe"));
    }
}
