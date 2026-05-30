package tech.kayys.gollek.models.layoutxlm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class LayoutXlmModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "layoutxlm",
                "LayoutXLM",
                List.of("layoutxlm"),
                List.of("LayoutXLMConfig", "LayoutLMv2Model",
                        "LayoutLMv2ForTokenClassification", "LayoutLMv2ForQuestionAnswering"),
                List.of(ModelFamilyCapability.MASKED_LM, ModelFamilyCapability.ENCODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.EMBEDDING,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/layoutxlm",
                        "tokenizer", "layoutxlm_sentencepiece_bpe_metadata_only",
                        "processor", "layoutxlm_processor",
                        "direct_safetensor", "pending_layoutxlm_processor_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "layoutxlm-sentencepiece-bpe",
                ModelTokenizerKind.CUSTOM,
                List.of(
                        List.of("tokenizer.json"),
                        List.of("tokenizer/tokenizer.json"),
                        List.of("sentencepiece.bpe.model"),
                        List.of("tokenizer/sentencepiece.bpe.model")),
                Map.of(
                        "pre_tokenizer", "sentencepiece_bpe",
                        "requires", "bbox_or_ocr_layout",
                        "status", "metadata_only_until_layoutxlm_processor_runtime")));
    }
}
