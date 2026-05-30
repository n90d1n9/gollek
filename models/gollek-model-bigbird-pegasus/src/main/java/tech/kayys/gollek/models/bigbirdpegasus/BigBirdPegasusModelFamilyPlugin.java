package tech.kayys.gollek.models.bigbirdpegasus;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class BigBirdPegasusModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "bigbird_pegasus",
                "BigBird-Pegasus",
                List.of("bigbird_pegasus", "bigbird-pegasus"),
                List.of("BigBirdPegasusForConditionalGeneration", "BigBirdPegasusModel",
                        "BigBirdPegasusForCausalLM", "BigBirdPegasusForSequenceClassification",
                        "BigBirdPegasusForQuestionAnswering"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/bigbird_pegasus",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_bigbird_pegasus_block_sparse_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "bigbird-pegasus-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("spiece.model"), List.of("tokenizer.model"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
