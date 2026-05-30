package tech.kayys.gollek.models.metaclip2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MetaClip2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "metaclip_2",
                "MetaCLIP 2",
                List.of("metaclip_2", "metaclip_2_text_model", "metaclip_2_vision_model"),
                List.of("MetaClip2Model", "MetaClip2TextModel", "MetaClip2VisionModel",
                        "MetaClip2TextModelWithProjection", "MetaClip2VisionModelWithProjection",
                        "MetaClip2ForImageClassification"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/metaclip_2",
                        "tokenizer", "clip_byte_level_bpe",
                        "image_processor", "metaclip_2_image_processor",
                        "direct_safetensor", "pending_metaclip2_dual_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("metaclip-2-byte-level-bpe"));
    }
}
