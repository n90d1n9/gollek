package tech.kayys.gollek.models.mobilevitv2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MobileVitV2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mobilevitv2",
                "MobileViT v2",
                List.of("mobilevitv2", "mobilevit_v2", "mobilevit-v2"),
                List.of("MobileViTV2Model", "MobileViTV2ForImageClassification",
                        "MobileViTV2ForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/mobilevitv2",
                        "image_processor", "mobilevitv2_image_processor",
                        "direct_safetensor", "pending_mobilevitv2_separable_self_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
