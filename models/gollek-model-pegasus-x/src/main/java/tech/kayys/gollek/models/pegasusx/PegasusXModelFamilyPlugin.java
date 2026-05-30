package tech.kayys.gollek.models.pegasusx;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class PegasusXModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "pegasus_x",
                "Pegasus-X",
                List.of("pegasus_x", "pegasus-x"),
                List.of("PegasusXForConditionalGeneration", "PegasusXModel",
                        "PegasusXEncoder", "PegasusXDecoder"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/pegasus_x",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_pegasus_x_encoder_decoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "pegasus-x-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("spiece.model"),
                        List.of("tokenizer.model"),
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
