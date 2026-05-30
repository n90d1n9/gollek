package tech.kayys.gollek.models;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyContractValidator;
import tech.kayys.gollek.spi.model.ModelFamilyContractViolation;
import tech.kayys.gollek.spi.model.ModelFamilyFixtureValidator;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GemmaModelFamilyPluginTest {

    @Test
    void gemmaPluginSatisfiesSharedModelFamilyContract() {
        GemmaModelFamilyPlugin plugin = new GemmaModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void gemmaFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new GemmaModelFamilyPlugin(),
                fixture("gemma"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "gemma fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesGemmaDirectArchitectureAdapter() {
        GemmaModelFamilyPlugin plugin = new GemmaModelFamilyPlugin();

        assertEquals(List.of("gemma"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof GemmaFamily);
        assertTrue(plugin.descriptor().modelTypes().contains("gemma"));
        assertTrue(plugin.descriptor().architectureClassNames().contains("GemmaForCausalLM"));
    }

    @Test
    void gemmaDirectWeightsExposeCoreAndGemmaSpecificLayout() {
        GemmaFamily architecture = new GemmaFamily();

        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.embed_tokens_per_layer.weight", architecture.embedTokensPerLayerWeight());
        assertEquals("model.per_layer_model_projection.weight", architecture.perLayerModelProjectionWeight());
        assertEquals("model.per_layer_projection_norm.weight", architecture.perLayerProjectionNormWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals("model.layers.6.self_attn.q_proj.weight", architecture.layerQueryWeight(6));
        assertEquals("model.layers.6.self_attn.k_proj.weight", architecture.layerKeyWeight(6));
        assertEquals("model.layers.6.self_attn.v_proj.weight", architecture.layerValueWeight(6));
        assertEquals("model.layers.6.self_attn.o_proj.weight", architecture.layerOutputWeight(6));
        assertEquals("model.layers.6.input_layernorm.weight", architecture.layerAttentionNormWeight(6));
        assertEquals("model.layers.6.self_attn.q_norm.weight", architecture.layerQueryNormWeight(6));
        assertEquals("model.layers.6.self_attn.k_norm.weight", architecture.layerKeyNormWeight(6));
        assertEquals("model.layers.6.post_attention_layernorm.weight", architecture.layerPostAttnNormWeight(6));
        assertEquals("model.layers.6.pre_feedforward_layernorm.weight", architecture.layerPreFfnNormWeight(6));
        assertEquals("model.layers.6.post_feedforward_layernorm.weight", architecture.layerFfnNormWeight(6));
        assertEquals("model.layers.6.post_feedforward_layernorm.weight", architecture.layerPostFfnNormWeight(6));
    }

    @Test
    void gemmaDirectWeightsExposeGemma4PerLayerInputs() {
        GemmaFamily architecture = new GemmaFamily();

        assertEquals("model.layers.3.per_layer_input_gate.weight", architecture.layerPerLayerInputGateWeight(3));
        assertEquals("model.layers.3.per_layer_projection.weight", architecture.layerPerLayerProjectionWeight(3));
        assertEquals("model.layers.3.post_per_layer_input_norm.weight", architecture.layerPostPerLayerInputNormWeight(3));
        assertEquals("model.layers.3.layer_scalar", architecture.layerScalarWeight(3));
    }

    @Test
    void gemmaRuntimeTraitsUseGeluNeoxRopeAndScaledEmbeddings() {
        GemmaFamily architecture = new GemmaFamily();

        assertEquals(FFNActivationType.GELU, architecture.activationType());
        assertTrue(architecture.usesNeoxRope());
        assertTrue(architecture.addOneToRmsNormWeight());
        assertEquals(64.0f, architecture.embeddingScaleFactor(4096));
        assertEquals(0.0f, architecture.defaultAttnSoftCap());
        assertEquals(0.0f, architecture.defaultFinalSoftCap());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                GemmaModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
