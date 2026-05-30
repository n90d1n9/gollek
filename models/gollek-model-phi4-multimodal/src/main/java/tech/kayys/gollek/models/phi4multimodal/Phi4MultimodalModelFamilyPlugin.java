package tech.kayys.gollek.models.phi4multimodal;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class Phi4MultimodalModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "phi4_multimodal",
                "Microsoft Phi-4 Multimodal",
                List.of("phi4_multimodal", "phi4_multimodal_vision", "phi4_multimodal_audio"),
                List.of("Phi4MultimodalForCausalLM", "Phi4MultimodalModel",
                        "Phi4MultimodalVisionModel", "Phi4MultimodalAudioModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.AUDIO, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/phi4_multimodal",
                        "tokenizer", "phi_hf_bpe_with_audio_vision_processor",
                        "direct_safetensor", "pending_phi4_multimodal_projectors_and_audio_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "phi4-multimodal-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_phi4_multimodal",
                        "image_processor", "image_processing_phi4_multimodal",
                        "feature_extractor", "feature_extraction_phi4_multimodal",
                        "status", "metadata_only")));
    }
}
