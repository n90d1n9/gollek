package tech.kayys.gollek.models.encodec;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class EncodecModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "encodec",
                "EnCodec",
                List.of("encodec"),
                List.of("EncodecModel"),
                List.of(ModelFamilyCapability.AUDIO, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.DECODER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/encodec",
                        "feature_extractor", "encodec_feature_extractor",
                        "direct_safetensor", "pending_neural_audio_codec_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }
}
