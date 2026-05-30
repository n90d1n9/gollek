package tech.kayys.gollek.models.resnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ResNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "resnet",
                "ResNet",
                List.of("resnet"),
                List.of("ResNetModel", "ResNetForImageClassification", "ResNetBackbone"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/resnet",
                        "image_processor", "resnet_image_processor",
                        "direct_safetensor", "not_causal_lm_image_backbone_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
