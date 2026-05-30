package tech.kayys.gollek.models.levit;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LevitModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "levit",
                "LeViT",
                List.of("levit"),
                List.of("LevitModel", "LevitForImageClassification",
                        "LevitForImageClassificationWithTeacher"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/levit",
                        "image_processor", "levit_image_processor",
                        "direct_safetensor", "pending_levit_hybrid_cnn_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
