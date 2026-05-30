package tech.kayys.gollek.models.luke;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LukeModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "luke",
                "LUKE",
                List.of("luke"),
                List.of("LukeModel", "LukeForMaskedLM", "LukeForEntityClassification",
                        "LukeForEntityPairClassification", "LukeForEntitySpanClassification",
                        "LukeForSequenceClassification", "LukeForTokenClassification",
                        "LukeForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/luke",
                        "tokenizer", "byte_level_bpe_with_entity_vocab",
                        "direct_safetensor", "pending_entity_aware_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "luke-byte-level-bpe-entity-vocab",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.json", "merges.txt", "entity_vocab.json"),
                        List.of("tokenizer/vocab.json", "tokenizer/merges.txt",
                                "tokenizer/entity_vocab.json")),
                Map.of(
                        "pre_tokenizer", "byte_level_bpe",
                        "status", "metadata_only_until_entity_vocab_runtime")));
    }
}
