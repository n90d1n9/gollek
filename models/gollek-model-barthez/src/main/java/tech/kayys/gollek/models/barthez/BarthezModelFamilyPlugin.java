package tech.kayys.gollek.models.barthez;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BarthezModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "barthez",
                "BARThez",
                List.of("barthez"),
                List.of("BartForConditionalGeneration", "BartModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/barthez",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_bart_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "barthez-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("sentencepiece.bpe.model"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("tokenizer/sentencepiece.bpe.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
