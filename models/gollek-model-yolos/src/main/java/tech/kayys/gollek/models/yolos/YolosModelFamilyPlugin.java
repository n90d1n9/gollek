package tech.kayys.gollek.models.yolos;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class YolosModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "yolos",
                "YOLOS",
                List.of("yolos"),
                List.of("YolosModel", "YolosForObjectDetection"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/yolos",
                        "image_processor", "yolos_image_processor",
                        "direct_safetensor", "pending_yolos_detection_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
