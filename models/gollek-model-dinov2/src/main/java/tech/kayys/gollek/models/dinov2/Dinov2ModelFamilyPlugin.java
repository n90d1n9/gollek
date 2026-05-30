package tech.kayys.gollek.models.dinov2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Dinov2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "dinov2",
                "DINOv2",
                List.of("dinov2"),
                List.of("Dinov2Model", "Dinov2ForImageClassification", "Dinov2Backbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/dinov2",
                        "image_processor", "vit_image_processor",
                        "direct_safetensor", "pending_dinov2_backbone_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
