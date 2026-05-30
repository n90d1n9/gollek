package tech.kayys.gollek.models.flava;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FlavaModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "flava",
                "Meta FLAVA",
                List.of("flava", "flava_text_model", "flava_image_model", "flava_multimodal_model"),
                List.of("FlavaModel", "FlavaForPreTraining", "FlavaTextModel",
                        "FlavaImageModel", "FlavaMultimodalModel"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/flava",
                        "tokenizer", "wordpiece",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("flava-wordpiece"));
    }
}
