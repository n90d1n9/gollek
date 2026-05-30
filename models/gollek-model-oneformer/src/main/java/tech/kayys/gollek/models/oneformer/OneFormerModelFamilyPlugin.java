package tech.kayys.gollek.models.oneformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class OneFormerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "oneformer",
                "OneFormer",
                List.of("oneformer"),
                List.of("OneFormerModel", "OneFormerForUniversalSegmentation"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/oneformer",
                        "tokenizer", "wordpiece_task_text",
                        "image_processor", "oneformer_image_processor",
                        "direct_safetensor", "pending_universal_segmentation_text_task_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("oneformer-task-wordpiece"));
    }
}
