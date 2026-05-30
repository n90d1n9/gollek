package tech.kayys.gollek.models.gemma3;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyContractValidator;
import tech.kayys.gollek.spi.model.ModelFamilyContractViolation;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Gemma3ModelFamilyPluginTest {

    @Test
    void gemma3PluginSatisfiesSharedModelFamilyContract() {
        Gemma3ModelFamilyPlugin plugin = new Gemma3ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma3 model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void publishesGemma3TextDirectArchitectureAdapterAndTokenizer() {
        Gemma3ModelFamilyPlugin plugin = new Gemma3ModelFamilyPlugin();

        assertEquals(List.of("gemma3_text"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Gemma3TextFamily);
        assertEquals(List.of("gemma3", "gemma3_text"), plugin.descriptor().modelTypes());
        assertEquals(List.of("Gemma3ForCausalLM"), plugin.descriptor().architectureClassNames());
        assertEquals("text_adapter_only", plugin.descriptor().metadata().get("direct_safetensor"));
        assertEquals("Gemma3ForConditionalGeneration", plugin.descriptor().metadata().get("multimodal_architecture"));
        assertEquals(List.of("gemma3-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void gemma3TextDirectWeightsExposeQkNormAndGemmaNormLayout() {
        Gemma3TextFamily architecture = new Gemma3TextFamily();

        assertEquals(List.of("Gemma3ForCausalLM"), architecture.supportedArchClassNames());
        assertEquals(List.of("gemma3", "gemma3_text"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.embed_tokens.weight", architecture.lmHeadWeight());
        assertTrue(architecture.hasTiedEmbeddings());
        assertEquals("model.layers.5.self_attn.q_proj.weight", architecture.layerQueryWeight(5));
        assertEquals("model.layers.5.self_attn.k_proj.weight", architecture.layerKeyWeight(5));
        assertEquals("model.layers.5.self_attn.v_proj.weight", architecture.layerValueWeight(5));
        assertEquals("model.layers.5.self_attn.o_proj.weight", architecture.layerOutputWeight(5));
        assertEquals("model.layers.5.self_attn.q_norm.weight", architecture.layerQueryNormWeight(5));
        assertEquals("model.layers.5.self_attn.k_norm.weight", architecture.layerKeyNormWeight(5));
        assertEquals(architecture.layerQueryNormWeight(5), architecture.layerQNorm(5));
        assertEquals(architecture.layerKeyNormWeight(5), architecture.layerKNorm(5));
        assertEquals("model.layers.5.input_layernorm.weight", architecture.layerAttentionNormWeight(5));
        assertEquals("model.layers.5.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(5));
        assertEquals(architecture.layerPostAttnNormWeight(5), architecture.layerPostAttnNorm(5));
        assertEquals("model.layers.5.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(5));
        assertEquals("model.layers.5.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(5));
        assertEquals("model.layers.5.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(5));
    }

    @Test
    void gemma3TextRuntimeTraitsExposeTextAdapterBehavior() {
        Gemma3TextFamily architecture = new Gemma3TextFamily();
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.GEMMA_TURN_AWARE, traits.promptBosPolicy());
        assertTrue(traits.gemma3Text());
    }
}
