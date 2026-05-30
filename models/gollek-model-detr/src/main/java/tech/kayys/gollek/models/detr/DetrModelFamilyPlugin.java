package tech.kayys.gollek.models.detr;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DetrModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "detr",
                "DETR / Deformable DETR",
                List.of("detr", "conditional_detr", "conditional-detr",
                        "deformable_detr", "deformable-detr", "rt_detr", "rt-detr"),
                List.of("DetrModel", "DetrForObjectDetection", "DetrForSegmentation",
                        "ConditionalDetrModel", "ConditionalDetrForObjectDetection",
                        "DeformableDetrModel", "DeformableDetrForObjectDetection",
                        "RTDetrModel", "RTDetrForObjectDetection"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/detr",
                        "image_processor", "detr_image_processor",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
