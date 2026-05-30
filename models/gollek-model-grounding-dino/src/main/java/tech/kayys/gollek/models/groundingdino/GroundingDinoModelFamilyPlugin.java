package tech.kayys.gollek.models.groundingdino;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class GroundingDinoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "grounding_dino",
                "Grounding DINO / MM Grounding DINO",
                List.of("grounding_dino", "grounding-dino",
                        "mm_grounding_dino", "mm-grounding-dino"),
                List.of("GroundingDinoModel", "GroundingDinoForObjectDetection",
                        "MMGroundingDinoModel", "MMGroundingDinoForObjectDetection"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/grounding_dino,mm_grounding_dino",
                        "tokenizer", "wordpiece",
                        "image_processor", "grounding_dino_image_processor",
                        "direct_safetensor", "pending_multiscale_deformable_attention_detection_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("grounding-dino-wordpiece"));
    }
}
