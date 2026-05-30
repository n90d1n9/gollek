package tech.kayys.gollek.models.plbart;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PLBartModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "plbart",
                "PLBART",
                List.of("plbart"),
                List.of("PLBartForConditionalGeneration", "PLBartModel",
                        "PLBartForSequenceClassification", "PLBartForCausalLM"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/plbart",
                        "tokenizer", "sentencepiece_programming_language_bart",
                        "direct_safetensor", "pending_plbart_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "plbart-sentencepiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("sentencepiece.bpe.model"), List.of("tokenizer.model"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_plbart",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
