package tech.kayys.gollek.models.wavlm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class WavLMModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "wavlm",
                "WavLM",
                List.of("wavlm"),
                List.of("WavLMModel", "WavLMForCTC", "WavLMForSequenceClassification"),
                List.of(ModelFamilyCapability.AUDIO, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/wavlm",
                        "feature_extractor", "wav2vec2_feature_extractor",
                        "tokenizer", "ctc_vocab_metadata_only",
                        "direct_safetensor", "pending_audio_feature_extractor_and_ctc_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "wavlm-ctc-vocab",
                ModelTokenizerKind.CUSTOM,
                List.of(),
                Map.of(
                        "decoder", "ctc",
                        "status", "metadata_only_until_ctc_tokenizer_runtime")));
    }
}
