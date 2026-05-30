package tech.kayys.gollek.models.gemma4;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;
import static java.util.Map.entry;

@ApplicationScoped
public class Gemma4ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "gemma4",
                "Google Gemma 4",
                List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio"),
                List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration",
                        "Gemma4Model", "Gemma4TextModel", "Gemma4VisionModel",
                        "Gemma4AudioModel", "Gemma4Processor", "Gemma4ImageProcessor",
                        "Gemma4VideoProcessor"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.AUDIO,
                        ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE, ModelFamilyCapability.GGUF),
                Map.ofEntries(
                        entry("bundle_profile", "optional"),
                        entry("origin", "3rdparty/transformers/src/transformers/models/gemma4"),
                        entry("direct_safetensor", "experimental_text_path_guarded_by_runtime"),
                        entry("direct_safetensor_scope", "text_only_gemma4_text"),
                        entry("moe_direct_safetensor", "pending_packed_expert_router_runtime"),
                        entry("double_wide_mlp", "detected_from_text_config"),
                        entry("multimodal_direct_safetensor", "pending_audio_vision_video_embedder_runtime"),
                        entry("tokenizer", "gemma_sentencepiece_with_audio_vision_processor"),
                        entry("processor", "Gemma4Processor"),
                        entry("image_processor", "Gemma4ImageProcessor"),
                        entry("video_processor", "Gemma4VideoProcessor"),
                        entry("feature_extractor", "Gemma4FeatureExtractor"),
                        entry("version", "0.1.0-SNAPSHOT")));
    }

    @Override
    public List<ModelArchitecture> architectureAdapters() {
        return List.of(new Gemma4Family());
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("gemma4-spm-bpe"));
    }
}
