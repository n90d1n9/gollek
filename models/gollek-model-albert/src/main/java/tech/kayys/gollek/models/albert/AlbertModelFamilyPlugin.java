package tech.kayys.gollek.models.albert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class AlbertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "albert",
                "ALBERT",
                List.of("albert"),
                List.of("AlbertModel", "AlbertForMaskedLM", "AlbertForSequenceClassification",
                        "AlbertForTokenClassification", "AlbertForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/albert",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "not_causal_lm",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "albert-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
