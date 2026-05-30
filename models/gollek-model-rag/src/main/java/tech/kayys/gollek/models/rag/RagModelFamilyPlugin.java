package tech.kayys.gollek.models.rag;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class RagModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "rag",
                "Retrieval-Augmented Generation",
                List.of("rag"),
                List.of("RagModel", "RagSequenceForGeneration", "RagTokenForGeneration"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/rag",
                        "tokenizer", "composite_question_encoder_generator_tokenizer",
                        "direct_safetensor", "pending_retriever_index_and_generator_composition_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "rag-composite-tokenizer",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("question_encoder_tokenizer", "generator_tokenizer"),
                        List.of("tokenizer.json")),
                Map.of(
                        "tokenizer", "tokenization_rag",
                        "retriever", "retrieval_rag",
                        "status", "metadata_only")));
    }
}
