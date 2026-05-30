package tech.kayys.gollek.models.xlmrobertxl;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class XlmRobertaXlModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "xlm_roberta_xl",
                "XLM-RoBERTa-XL",
                List.of("xlm-roberta-xl", "xlm_roberta_xl"),
                List.of("XLMRobertaXLModel", "XLMRobertaXLForMaskedLM",
                        "XLMRobertaXLForCausalLM",
                        "XLMRobertaXLForSequenceClassification",
                        "XLMRobertaXLForTokenClassification",
                        "XLMRobertaXLForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/xlm_roberta_xl",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_xlm_roberta_xl_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "xlm-roberta-xl-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
