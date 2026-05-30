package tech.kayys.gollek.models.vilt;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ViltModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "vilt",
                "ViLT",
                List.of("vilt"),
                List.of("ViltModel", "ViltForMaskedLM", "ViltForQuestionAnswering",
                        "ViltForImageAndTextRetrieval", "ViltForImagesAndTextClassification",
                        "ViltForTokenClassification"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.VISION,
                        ModelFamilyCapability.MULTIMODAL, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/vilt",
                        "tokenizer", "wordpiece_with_vilt_processor",
                        "image_processor", "vilt_image_processor",
                        "direct_safetensor", "pending_vilt_patch_embedding_and_multimodal_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "vilt-processor-wordpiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("vocab.txt"), List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "processor", "processing_vilt",
                        "image_processor", "image_processing_vilt",
                        "tokenizer", "wordpiece",
                        "status", "metadata_only")));
    }
}
