package tech.kayys.gollek.models.segformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SegformerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "segformer",
                "SegFormer",
                List.of("segformer"),
                List.of("SegformerModel", "SegformerForImageClassification",
                        "SegformerForSemanticSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/segformer",
                        "image_processor", "segformer_image_processor",
                        "direct_safetensor", "pending_segformer_decode_head_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
