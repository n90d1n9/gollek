package tech.kayys.gollek.models.siglip2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Siglip2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "siglip2",
                "SigLIP2",
                List.of("siglip2", "siglip2_text_model", "siglip2_vision_model"),
                List.of("Siglip2Model", "Siglip2TextModel", "Siglip2VisionModel",
                        "Siglip2ForImageClassification"),
                List.of(ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/siglip2",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "image_processor", "siglip2_image_processor",
                        "direct_safetensor", "not_causal_lm_siglip2_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "siglip2-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("spm.model"),
                        List.of("tokenizer.model"),
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
