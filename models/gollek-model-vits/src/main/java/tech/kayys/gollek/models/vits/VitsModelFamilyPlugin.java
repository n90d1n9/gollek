package tech.kayys.gollek.models.vits;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class VitsModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vits",
                "VITS",
                List.of("vits"),
                List.of("VitsModel"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.AUDIO,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/vits",
                        "tokenizer", "vits_text_tokenizer",
                        "direct_safetensor", "pending_vits_duration_flow_decoder_vocoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "vits-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.json"), List.of("tokenizer/vocab.json"), List.of("tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_vits",
                        "status", "metadata_only")));
    }
}
