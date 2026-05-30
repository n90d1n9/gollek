package tech.kayys.gollek.models.maskformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MaskFormerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "maskformer",
                "MaskFormer / Mask2Former",
                List.of("maskformer", "mask2former", "mask2-former"),
                List.of("MaskFormerModel", "MaskFormerForInstanceSegmentation",
                        "MaskFormerSwinModel", "MaskFormerSwinBackbone",
                        "Mask2FormerModel", "Mask2FormerForUniversalSegmentation"),
                List.of(ModelFamilyCapability.VISION, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/maskformer,mask2former",
                        "image_processor", "maskformer_image_processor",
                        "direct_safetensor", "pending_mask_decoder_panoptic_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
