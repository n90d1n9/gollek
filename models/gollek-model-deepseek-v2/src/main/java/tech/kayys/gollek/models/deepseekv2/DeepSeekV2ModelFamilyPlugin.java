package tech.kayys.gollek.models.deepseekv2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DeepSeekV2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "deepseek_v2",
                "DeepSeek-V2",
                List.of("deepseek_v2", "deepseek_v2_moe"),
                List.of("DeepseekV2ForCausalLM", "DeepseekV2Model",
                        "DeepseekV2ForSequenceClassification"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.GGUF),
                Map.of(
                        "bundle_profile", "experimental",
                        "origin", "3rdparty/transformers/src/transformers/models/deepseek_v2",
                        "direct_safetensor", "pending_deepseek_v2_mla_sparse_moe_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("deepseek-v2-hf-bpe"));
    }
}
