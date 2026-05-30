package tech.kayys.gollek.models.deepseekvlhybrid;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DeepSeekVlHybridModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deepseek_vl_hybrid",
                "DeepSeek-VL Hybrid",
                List.of("deepseek_vl_hybrid"),
                List.of("DeepseekVLHybridForConditionalGeneration", "DeepseekVLHybridModel",
                        "DeepseekVLHybridPreTrainedModel"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "experimental",
                        "origin", "3rdparty/transformers/src/transformers/models/deepseek_vl_hybrid",
                        "tokenizer", "deepseek_hf_bpe_with_vl_hybrid_processor",
                        "direct_safetensor", "pending_deepseek_vl_hybrid_sam_vision_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "deepseek-vl-hybrid-processor-tokenizer-json",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_deepseek_vl_hybrid",
                        "image_processor", "image_processing_deepseek_vl_hybrid",
                        "status", "experimental")));
    }
}
