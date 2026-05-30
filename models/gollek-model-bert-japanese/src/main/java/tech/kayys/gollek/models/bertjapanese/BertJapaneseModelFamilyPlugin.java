package tech.kayys.gollek.models.bertjapanese;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BertJapaneseModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bert_japanese",
                "BERT Japanese",
                List.of("bert_japanese", "bert-japanese"),
                List.of("BertModel", "BertForMaskedLM", "BertForSequenceClassification",
                        "BertForTokenClassification", "BertForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/bert_japanese",
                        "tokenizer", "japanese_wordpiece_or_sentencepiece",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_bert_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bert-japanese-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("vocab.txt"),
                        List.of("spiece.model"), List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer/vocab.txt"), List.of("tokenizer/spiece.model")),
                Map.of(
                        "word_tokenizer", "basic_mecab_sudachi_or_jumanpp",
                        "subword_tokenizer", "wordpiece_character_or_sentencepiece",
                        "status", "metadata_only_until_japanese_segmentation_runtime")));
    }
}
