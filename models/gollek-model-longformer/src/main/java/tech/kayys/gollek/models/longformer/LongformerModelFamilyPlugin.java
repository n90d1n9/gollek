package tech.kayys.gollek.models.longformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LongformerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "longformer",
                "Longformer / LED",
                List.of("longformer", "led"),
                List.of("LongformerModel", "LongformerForMaskedLM",
                        "LongformerForSequenceClassification", "LongformerForQuestionAnswering",
                        "LongformerForTokenClassification", "LEDModel",
                        "LEDForConditionalGeneration", "LEDForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.DECODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/longformer,led",
                        "tokenizer", "byte_level_bpe",
                        "direct_safetensor", "pending_sliding_window_global_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.huggingFaceBpe("longformer-byte-level-bpe"));
    }
}
