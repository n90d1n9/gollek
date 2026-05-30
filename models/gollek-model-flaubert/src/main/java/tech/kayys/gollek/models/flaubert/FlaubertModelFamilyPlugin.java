package tech.kayys.gollek.models.flaubert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class FlaubertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "flaubert",
                "FlauBERT",
                List.of("flaubert"),
                List.of("FlaubertModel", "FlaubertWithLMHeadModel",
                        "FlaubertForSequenceClassification",
                        "FlaubertForTokenClassification", "FlaubertForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/flaubert",
                        "tokenizer", "moses_bpe_metadata_only",
                        "direct_safetensor", "pending_xlm_style_encoder_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "flaubert-moses-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("vocab.json", "merges.txt"),
                        List.of("tokenizer/vocab.json", "tokenizer/merges.txt")),
                Map.of(
                        "pre_tokenizer", "moses",
                        "status", "metadata_only_until_moses_bpe_runtime")));
    }
}
