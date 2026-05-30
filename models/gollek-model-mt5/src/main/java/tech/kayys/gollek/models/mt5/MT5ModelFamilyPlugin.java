package tech.kayys.gollek.models.mt5;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class MT5ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "mt5",
                "mT5",
                List.of("mt5"),
                List.of("MT5ForConditionalGeneration", "MT5Model", "MT5EncoderModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/mt5",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_mt5_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(sentencePieceUnigram("mt5-sentencepiece-unigram"));
    }

    private static ModelTokenizerDescriptor sentencePieceUnigram(String id) {
        return new ModelTokenizerDescriptor(
                id,
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("spiece.model"),
                        List.of("sentencepiece.model"), List.of("tokenizer.model"),
                        List.of("tokenizer/tokenizer.json"), List.of("tokenizer/spiece.model"),
                        List.of("tokenizer/sentencepiece.model"), List.of("tokenizer/tokenizer.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime"));
    }
}
