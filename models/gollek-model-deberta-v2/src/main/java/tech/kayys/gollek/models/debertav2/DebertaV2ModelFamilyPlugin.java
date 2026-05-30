package tech.kayys.gollek.models.debertav2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DebertaV2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deberta_v2",
                "DeBERTa-v2",
                List.of("deberta_v2", "deberta-v2"),
                List.of("DebertaV2Model", "DebertaV2ForMaskedLM",
                        "DebertaV2ForSequenceClassification", "DebertaV2ForTokenClassification",
                        "DebertaV2ForQuestionAnswering", "DebertaV2ForMultipleChoice"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/deberta_v2",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "not_causal_lm_deberta_v2_encoder_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "deberta-v2-sentencepiece-unigram",
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
