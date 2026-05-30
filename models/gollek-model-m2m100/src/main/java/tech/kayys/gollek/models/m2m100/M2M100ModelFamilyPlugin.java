package tech.kayys.gollek.models.m2m100;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class M2M100ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "m2m100",
                "Meta M2M100",
                List.of("m2m_100", "m2m100"),
                List.of("M2M100ForConditionalGeneration", "M2M100Model",
                        "M2M100Encoder", "M2M100Decoder"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/m2m_100",
                        "tokenizer", "sentencepiece_multilingual_m2m100",
                        "direct_safetensor", "pending_m2m100_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "m2m100-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("sentencepiece.bpe.model"), List.of("tokenizer.model"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_m2m_100",
                        "language_codes", "required",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
