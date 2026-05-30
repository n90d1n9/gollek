package tech.kayys.gollek.models.bartpho;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BartphoModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bartpho",
                "BARTpho",
                List.of("bartpho"),
                List.of("BartForConditionalGeneration", "BartModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/bartpho",
                        "tokenizer", "sentencepiece_bpe_with_monolingual_vocab",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only_over_bart_backbone",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bartpho-sentencepiece-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("sentencepiece.bpe.model", "dict.txt"),
                        List.of("tokenizer/sentencepiece.bpe.model", "tokenizer/dict.txt")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_bpe",
                        "status", "metadata_only_until_bartpho_tokenizer_runtime")));
    }
}
