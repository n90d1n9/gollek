package tech.kayys.gollek.models.rembert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RemBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "rembert",
                "RemBERT",
                List.of("rembert"),
                List.of("RemBertModel", "RemBertForMaskedLM", "RemBertForCausalLM",
                        "RemBertForSequenceClassification", "RemBertForTokenClassification",
                        "RemBertForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/rembert",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_rembert_unigram_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "rembert-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("sentencepiece.model"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer/sentencepiece.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
