package tech.kayys.gollek.models.canine;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class CanineModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "canine",
                "CANINE",
                List.of("canine"),
                List.of("CanineModel", "CanineForSequenceClassification",
                        "CanineForTokenClassification", "CanineForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/canine",
                        "tokenizer", "unicode_codepoint_metadata_only",
                        "direct_safetensor", "pending_character_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "canine-unicode-codepoint",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "pre_tokenizer", "unicode_codepoint",
                        "status", "metadata_only_until_codepoint_tokenizer_runtime")));
    }
}
