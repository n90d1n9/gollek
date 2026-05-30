package tech.kayys.gollek.models.llava;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LlavaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "llava",
                "LLaVA",
                List.of("llava"),
                List.of("LlavaForConditionalGeneration", "LlavaModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/llava",
                        "tokenizer", "delegated_text_backbone_bpe_or_sentencepiece",
                        "image_processor", "llava_image_processor",
                        "direct_safetensor", "pending_vision_projector_and_text_backbone_composition_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(
                ModelTokenizerDescriptor.huggingFaceBpe("llava-hf-bpe"),
                ModelTokenizerDescriptor.sentencePieceBpe("llava-spm-bpe"));
    }
}
