package tech.kayys.gollek.models.groupvit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GroupVitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "groupvit",
                "GroupViT",
                List.of("groupvit", "groupvit_text_model", "groupvit_vision_model"),
                List.of("GroupViTModel", "GroupViTTextModel", "GroupViTVisionModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/groupvit",
                        "tokenizer", "clip_byte_level_bpe",
                        "image_processor", "groupvit_image_processor",
                        "direct_safetensor", "pending_groupvit_segmentation_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("groupvit-byte-level-bpe"));
    }
}
