package tech.kayys.gollek.models.whisper;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class WhisperModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "whisper",
                "OpenAI Whisper",
                List.of("whisper"),
                List.of("WhisperForConditionalGeneration", "WhisperModel",
                        "WhisperEncoder", "WhisperForAudioClassification"),
                List.of(ModelFamilyCapability.AUDIO, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.DECODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/whisper",
                        "tokenizer", "byte_level_bpe",
                        "direct_safetensor", "pending_audio_encoder_decoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("whisper-byte-level-bpe"));
    }
}
