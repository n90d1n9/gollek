package tech.kayys.gollek.models.instructblip;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class InstructBlipModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "instructblip",
                "InstructBLIP",
                List.of("instructblip", "instructblip_vision_model", "instructblip_qformer"),
                List.of("InstructBlipForConditionalGeneration", "InstructBlipModel",
                        "InstructBlipVisionModel", "InstructBlipQFormerModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.CHAT_TEMPLATE,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/instructblip",
                        "tokenizer", "qformer_processor_text_tokenizer",
                        "image_processor", "instructblip_vision_processor",
                        "direct_safetensor", "pending_blip_qformer_bridge_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "instructblip-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_instructblip",
                        "status", "metadata_only")));
    }
}
