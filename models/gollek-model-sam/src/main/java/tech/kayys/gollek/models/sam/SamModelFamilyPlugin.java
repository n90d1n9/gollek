package tech.kayys.gollek.models.sam;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SamModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "sam",
                "Segment Anything",
                List.of("sam", "sam_hq", "sam-hq", "sam2", "segment_anything"),
                List.of("SamModel", "SamHQModel", "Sam2Model"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/sam",
                        "image_processor", "sam_image_processor",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
