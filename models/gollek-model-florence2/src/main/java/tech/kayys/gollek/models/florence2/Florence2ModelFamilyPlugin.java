package tech.kayys.gollek.models.florence2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Florence2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "florence2",
                "Microsoft Florence-2",
                List.of("florence2", "florence_vision"),
                List.of("Florence2ForConditionalGeneration", "Florence2Model",
                        "Florence2VisionBackbone"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/florence2",
                        "tokenizer", "processor_backed_tokenizer_json",
                        "direct_safetensor", "pending_florence2_processor_vision_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "florence2-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_florence2",
                        "status", "metadata_only")));
    }
}
