package tech.kayys.gollek.models.regnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RegNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "regnet",
                "RegNet",
                List.of("regnet"),
                List.of("RegNetModel", "RegNetForImageClassification"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/regnet",
                        "image_processor", "regnet_image_processor",
                        "direct_safetensor", "pending_regnet_stage_runtime_validation",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
