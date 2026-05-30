package tech.kayys.gollek.models.focalnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FocalNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "focalnet",
                "FocalNet",
                List.of("focalnet"),
                List.of("FocalNetModel", "FocalNetForImageClassification",
                        "FocalNetForMaskedImageModeling", "FocalNetBackbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/focalnet",
                        "image_processor", "focalnet_image_processor",
                        "direct_safetensor", "pending_focal_modulation_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
