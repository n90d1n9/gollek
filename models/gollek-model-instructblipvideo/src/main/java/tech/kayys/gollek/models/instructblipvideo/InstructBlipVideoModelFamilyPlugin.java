package tech.kayys.gollek.models.instructblipvideo;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class InstructBlipVideoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "instructblipvideo",
                "InstructBLIP-Video",
                List.of("instructblipvideo", "instructblipvideo_vision_model",
                        "instructblipvideo_qformer"),
                List.of("InstructBlipVideoForConditionalGeneration", "InstructBlipVideoModel",
                        "InstructBlipVideoVisionModel", "InstructBlipVideoQFormerModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.CHAT_TEMPLATE,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/instructblipvideo",
                        "tokenizer", "qformer_processor_text_tokenizer",
                        "video_processor", "instructblipvideo_video_processor",
                        "direct_safetensor", "pending_instructblipvideo_qformer_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "instructblipvideo-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_instructblipvideo",
                        "status", "metadata_only")));
    }
}
