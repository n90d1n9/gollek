package tech.kayys.gollek.models.t5gemma2;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class T5Gemma2ModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "t5gemma2",
                "Google T5Gemma 2",
                List.of("t5gemma2", "t5gemma2_text", "t5gemma2_encoder", "t5gemma2_decoder"),
                List.of("T5Gemma2ForConditionalGeneration", "T5Gemma2Model", "T5Gemma2TextEncoder",
                        "T5Gemma2Encoder", "T5Gemma2Decoder", "T5Gemma2ForSequenceClassification",
                        "T5Gemma2ForTokenClassification"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.TOKENIZER, ModelFamilyCapability.MULTIMODAL,
                        ModelFamilyCapability.VISION, ModelFamilyCapability.TRAINING,
                        ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/t5gemma2",
                        "tokenizer", "gemma_sentencepiece_seq2seq_with_optional_vision_projector",
                        "direct_safetensor", "pending_t5gemma2_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(ModelTokenizerDescriptor.sentencePieceBpe("t5gemma2-spm-bpe"));
    }
}
