package tech.kayys.gollek.models.nllb;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class NllbModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "nllb",
                "Meta NLLB",
                List.of("nllb"),
                List.of("M2M100ForConditionalGeneration"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/nllb",
                        "tokenizer", "sentencepiece_multilingual_nllb",
                        "direct_safetensor", "pending_nllb_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "nllb-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("sentencepiece.bpe.model"), List.of("tokenizer.model"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_nllb",
                        "language_codes", "required",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
