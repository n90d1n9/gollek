package tech.kayys.gollek.models.prophetnet;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class ProphetNetModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "prophetnet",
                "ProphetNet",
                List.of("prophetnet", "xlm-prophetnet", "xlm_prophetnet"),
                List.of("ProphetNetForConditionalGeneration", "ProphetNetForCausalLM",
                        "ProphetNetModel", "XLMProphetNetForConditionalGeneration"),
                List.of(ModelFamilyCapability.ENCODER, ModelFamilyCapability.DECODER,
                        ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.TRAINING, ModelFamilyCapability.ONNX),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/prophetnet",
                        "tokenizer", "wordpiece_prophetnet",
                        "direct_safetensor", "pending_prophetnet_n_stream_seq2seq_runtime",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "prophetnet-wordpiece",
                ModelTokenizerKind.WORD_PIECE,
                List.of(List.of("prophetnet.tokenizer"),
                        List.of("tokenizer/prophetnet.tokenizer"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of("pre_tokenizer", "bert-basic")));
    }
}
