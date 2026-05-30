package tech.kayys.gollek.models.clipseg;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ClipSegModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "clipseg",
                "CLIPSeg",
                List.of("clipseg", "clipseg_text_model", "clipseg_vision_model"),
                List.of("CLIPSegModel", "CLIPSegTextModel", "CLIPSegVisionModel",
                        "CLIPSegForImageSegmentation", "CLIPSegDecoder"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/clipseg",
                        "tokenizer", "clip_byte_level_bpe",
                        "image_processor", "clipseg_processor",
                        "direct_safetensor", "pending_clipseg_decoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("clipseg-byte-level-bpe"));
    }
}
