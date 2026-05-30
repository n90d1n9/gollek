package tech.kayys.gollek.models.deepseekvl;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DeepSeekVlModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deepseek_vl",
                "DeepSeek-VL",
                List.of("deepseek_vl"),
                List.of("DeepseekVLForConditionalGeneration", "DeepseekVLModel",
                        "DeepseekVLPreTrainedModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "experimental",
                        "origin", "3rdparty/transformers/src/transformers/models/deepseek_vl",
                        "tokenizer", "deepseek_hf_bpe_with_vl_processor",
                        "direct_safetensor", "pending_deepseek_vl_vision_aligner_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "deepseek-vl-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_deepseek_vl",
                        "image_processor", "image_processing_deepseek_vl",
                        "status", "experimental")));
    }
}
