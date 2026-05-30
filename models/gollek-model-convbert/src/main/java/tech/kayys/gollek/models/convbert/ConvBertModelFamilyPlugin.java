package tech.kayys.gollek.models.convbert;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ConvBertModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "convbert",
                "ConvBERT",
                List.of("convbert"),
                List.of("ConvBertModel", "ConvBertForMaskedLM",
                        "ConvBertForSequenceClassification", "ConvBertForTokenClassification",
                        "ConvBertForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/convbert",
                        "tokenizer", "wordpiece",
                        "direct_safetensor", "pending_separable_conv_attention_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.wordPiece("convbert-wordpiece"));
    }
}
