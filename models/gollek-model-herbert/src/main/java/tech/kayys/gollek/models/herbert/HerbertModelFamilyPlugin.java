package tech.kayys.gollek.models.herbert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class HerbertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "herbert",
                "HerBERT",
                List.of("herbert"),
                List.of("RobertaModel", "RobertaForMaskedLM",
                        "RobertaForSequenceClassification", "RobertaForTokenClassification"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/herbert",
                        "tokenizer", "bert_pretokenized_bpe_metadata_only",
                        "direct_safetensor", "roberta_layout_tokenizer_runtime_pending",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "herbert-bert-pretokenized-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.json", "merges.txt"),
                        List.of("tokenizer/vocab.json", "tokenizer/merges.txt")),
                Map.of(
                        "pre_tokenizer", "bert_pre_tokenizer_bpe",
                        "status", "metadata_only_until_herbert_tokenizer_runtime")));
    }
}
