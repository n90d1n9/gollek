package tech.kayys.gollek.models.pixtral;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PixtralModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "pixtral",
                "Mistral Pixtral Vision",
                List.of("pixtral"),
                List.of("PixtralVisionModel", "PixtralPreTrainedModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/pixtral",
                        "tokenizer", "mistral_bpe_with_pixtral_processor",
                        "image_processor", "pixtral_image_processor",
                        "direct_safetensor", "pending_pixtral_vision_patch_projector_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "pixtral-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_pixtral",
                        "image_processor", "image_processing_pixtral",
                        "status", "metadata_only")));
    }
}
