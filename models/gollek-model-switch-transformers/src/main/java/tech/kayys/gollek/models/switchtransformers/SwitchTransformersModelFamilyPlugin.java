package tech.kayys.gollek.models.switchtransformers;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class SwitchTransformersModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "switch_transformers",
                "Switch Transformers",
                List.of("switch_transformers", "switch-transformers"),
                List.of("SwitchTransformersForConditionalGeneration", "SwitchTransformersModel",
                        "SwitchTransformersEncoderModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/switch_transformers",
                        "tokenizer", "sentencepiece_unigram_metadata_only",
                        "direct_safetensor", "pending_switch_transformers_moe_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "switch-transformers-sentencepiece-unigram",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("spiece.model"),
                        List.of("sentencepiece.model"), List.of("tokenizer.model"),
                        List.of("tokenizer/tokenizer.json"), List.of("tokenizer/spiece.model"),
                        List.of("tokenizer/sentencepiece.model"), List.of("tokenizer/tokenizer.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_unigram",
                        "status", "metadata_only_until_unigram_tokenizer_runtime")));
    }
}
