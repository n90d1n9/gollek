package tech.kayys.gollek.models.cpm;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;
import tech.kayys.gollek.spi.model.ModelTokenizerKind;

import java.util.List;
import java.util.Map;

@ApplicationScoped
public class CpmModelFamilyPlugin implements ModelFamilyPlugin {

    @Override
    public ModelFamilyDescriptor descriptor() {
        return new ModelFamilyDescriptor(
                "cpm",
                "CPM",
                List.of("cpm"),
                List.of("CpmTokenizer", "CpmTokenizerFast"),
                List.of(ModelFamilyCapability.CAUSAL_LM, ModelFamilyCapability.TOKENIZER,
                        ModelFamilyCapability.CHAT_TEMPLATE, ModelFamilyCapability.TRAINING),
                Map.of(
                        "bundle_profile", "metadata_only",
                        "origin", "3rdparty/transformers/src/transformers/models/cpm",
                        "tokenizer", "jieba_sentencepiece",
                        "direct_safetensor", "not_direct_runtime_tokenizer_only",
                        "version", "0.1.0-SNAPSHOT"));
    }

    @Override
    public List<ModelTokenizerDescriptor> tokenizerDescriptors() {
        return List.of(new ModelTokenizerDescriptor(
                "cpm-jieba-sentencepiece",
                ModelTokenizerKind.CUSTOM,
                List.of(List.of("spiece.model"), List.of("tokenizer/spiece.model"),
                        List.of("tokenizer.json"), List.of("tokenizer/tokenizer.json")),
                Map.of(
                        "pre_tokenizer", "rjieba",
                        "subword_tokenizer", "sentencepiece",
                        "status", "metadata_only_until_cpm_tokenizer_runtime")));
    }
}
