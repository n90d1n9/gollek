package tech.kayys.gollek.models.visiontextdualencoder;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VisionTextDualEncoderModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vision_text_dual_encoder",
                "Vision-Text Dual Encoder",
                List.of("vision_text_dual_encoder", "vision-text-dual-encoder"),
                List.of("VisionTextDualEncoderModel", "VisionTextDualEncoderProcessor"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/vision_text_dual_encoder",
                        "tokenizer", "delegated_text_model_tokenizer",
                        "image_processor", "delegated_vision_model_processor",
                        "direct_safetensor", "pending_composite_dual_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "vision-text-dual-encoder-delegated-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "processor", "VisionTextDualEncoderProcessor",
                        "status", "delegated_to_text_submodel")));
    }
}
