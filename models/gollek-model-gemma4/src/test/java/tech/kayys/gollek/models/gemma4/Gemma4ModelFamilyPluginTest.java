package tech.kayys.gollek.models.gemma4;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyContractValidator;
import tech.kayys.gollek.spi.model.ModelFamilyContractViolation;
import tech.kayys.gollek.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.gollek.spi.model.ModelRuntimeTraits;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Gemma4ModelFamilyPluginTest {

    @Test
    void gemma4PluginSatisfiesSharedModelFamilyContract() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemma4FixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new Gemma4ModelFamilyPlugin(),
                fixture("gemma4"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma4 fixture should match descriptor, tokenizer, and guarded direct adapter claims");
    }

    @Test
    void publishesGemma4TextDirectArchitectureAdapterAndTokenizer() {
        Gemma4ModelFamilyPlugin plugin = new Gemma4ModelFamilyPlugin();

        assertEquals(List.of("gemma4"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof Gemma4Family);
        assertEquals(List.of("gemma4", "gemma4_text", "gemma4_vision", "gemma4_audio"),
                plugin.descriptor().modelTypes());
        assertEquals("experimental_text_path_guarded_by_runtime",
                plugin.descriptor().metadata().get("direct_safetensor"));
        assertEquals("text_only_gemma4_text", plugin.descriptor().metadata().get("direct_safetensor_scope"));
        assertEquals("pending_audio_vision_video_embedder_runtime",
                plugin.descriptor().metadata().get("multimodal_direct_safetensor"));
        assertEquals(List.of("gemma4-spm-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void gemma4DirectWeightsExposePerLayerInputsAndQkNormLayout() {
        Gemma4Family architecture = new Gemma4Family();

        assertEquals(List.of("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"),
                architecture.supportedArchClassNames());
        assertEquals(List.of("gemma4", "gemma4_text"), architecture.supportedModelTypes());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.embed_tokens_per_layer.weight", architecture.embedTokensPerLayerWeight());
        assertEquals("model.per_layer_model_projection.weight", architecture.perLayerModelProjectionWeight());
        assertEquals("model.per_layer_projection_norm.weight", architecture.perLayerProjectionNormWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.layers.4.self_attn.q_proj.weight", architecture.layerQueryWeight(4));
        assertEquals("model.layers.4.self_attn.k_proj.weight", architecture.layerKeyWeight(4));
        assertEquals("model.layers.4.self_attn.v_proj.weight", architecture.layerValueWeight(4));
        assertEquals("model.layers.4.self_attn.o_proj.weight", architecture.layerOutputWeight(4));
        assertEquals("model.layers.4.self_attn.q_norm.weight", architecture.layerQueryNormWeight(4));
        assertEquals("model.layers.4.self_attn.k_norm.weight", architecture.layerKeyNormWeight(4));
        assertEquals("model.layers.4.input_layernorm.weight", architecture.layerAttentionNormWeight(4));
        assertEquals("model.layers.4.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(4));
        assertEquals("model.layers.4.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(4));
        assertEquals("model.layers.4.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(4));
        assertEquals("model.layers.4.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(4));
        assertEquals("model.layers.4.per_layer_input_gate.weight", architecture.layerPerLayerInputGateWeight(4));
        assertEquals("model.layers.4.per_layer_projection.weight", architecture.layerPerLayerProjectionWeight(4));
        assertEquals("model.layers.4.post_per_layer_input_norm.weight",
                architecture.layerPostPerLayerInputNormWeight(4));
        assertEquals("model.layers.4.layer_scalar", architecture.layerScalarWeight(4));
    }

    @Test
    void gemma4RuntimeTraitsExposeTextRuntimePoliciesWithoutConfig() {
        Gemma4Family architecture = new Gemma4Family();
        ModelRuntimeTraits traits = architecture.runtimeTraits(null);

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(0.0f, architecture.defaultAttnSoftCap());
        assertEquals(0.0f, architecture.defaultFinalSoftCap());
        assertTrue(!traits.gemma4Text());
        assertEquals(ModelRuntimeTraits.PromptBosPolicy.GEMMA_TURN_AWARE, traits.promptBosPolicy());
        assertTrue(traits.allowedControlTokenTexts().isEmpty());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                Gemma4ModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
