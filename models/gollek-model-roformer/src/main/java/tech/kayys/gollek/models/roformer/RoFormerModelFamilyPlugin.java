package tech.kayys.gollek.models.roformer;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RoFormerModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "roformer",
                "RoFormer",
                List.of("roformer"),
                List.of("RoFormerModel", "RoFormerForMaskedLM", "RoFormerForCausalLM",
                        "RoFormerForSequenceClassification", "RoFormerForTokenClassification",
                        "RoFormerForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.CAUSAL_LM,
                        ModelFamilyCapability.ENCODER, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.EMBEDDING, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/roformer",
                        "tokenizer", "jieba_wordpiece_metadata_only",
                        "direct_safetensor", "pending_roformer_rotary_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "roformer-jieba-wordpiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("tokenizer.json"), List.of("vocab.txt"),
                        List.of("tokenizer/tokenizer.json"), List.of("tokenizer/vocab.txt")),
                Map.of(
                        "pre_tokenizer", "jieba_wordpiece",
                        "status", "metadata_only_until_jieba_wordpiece_runtime")));
    }
}
