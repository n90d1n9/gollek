package tech.kayys.gollek.models.upernet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class UperNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "upernet",
                "UPerNet",
                List.of("upernet"),
                List.of("UperNetForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/upernet",
                        "image_processor", "semantic_segmentation_image_processor",
                        "direct_safetensor", "pending_pyramid_pooling_segmentation_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
