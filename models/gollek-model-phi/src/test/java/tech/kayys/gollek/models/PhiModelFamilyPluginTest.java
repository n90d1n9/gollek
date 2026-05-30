package tech.kayys.gollek.models;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelFamilyContractValidator;
import tech.kayys.gollek.spi.model.ModelFamilyContractViolation;
import tech.kayys.gollek.spi.model.ModelFamilyFixtureValidator;
import tech.kayys.gollek.spi.model.ModelTokenizerDescriptor;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PhiModelFamilyPluginTest {

    @Test
    void phiPluginSatisfiesSharedModelFamilyContract() {
        PhiModelFamilyPlugin plugin = new PhiModelFamilyPlugin();

        List<ModelFamilyContractViolation> violations = ModelFamilyContractValidator.validate(plugin);

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "phi model-family plugin should satisfy the shared plugin contract");
    }

    @Test
    void phiFixtureMatchesDescriptorTokenizerAndAdapter() throws Exception {
        List<ModelFamilyContractViolation> violations = ModelFamilyFixtureValidator.validate(
                new PhiModelFamilyPlugin(),
                fixture("phi"));

        assertEquals(List.of(), violations.stream()
                        .map(ModelFamilyContractViolation::summary)
                        .toList(),
                "phi fixture should match descriptor, tokenizer, and direct adapter claims");
    }

    @Test
    void publishesPhiDirectArchitectureAdapterAndTokenizers() {
        PhiModelFamilyPlugin plugin = new PhiModelFamilyPlugin();

        assertEquals(List.of("phi"), plugin.architectureAdapters().stream()
                .map(ModelArchitecture::id)
                .toList());
        assertTrue(plugin.architectureAdapters().get(0) instanceof PhiFamily);
        assertTrue(plugin.descriptor().modelTypes().containsAll(List.of("phi", "phi3")));
        assertTrue(plugin.descriptor().architectureClassNames()
                .containsAll(List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM")));
        assertEquals(List.of("phi-spm-bpe", "phi-hf-bpe"), plugin.tokenizerDescriptors().stream()
                .map(ModelTokenizerDescriptor::id)
                .toList());
    }

    @Test
    void phiDirectWeightsExposeFusedQkvAndFusedGateUpLayout() {
        PhiFamily architecture = new PhiFamily();

        assertTrue(architecture.hasFusedQKV());
        assertEquals("model.embed_tokens.weight", architecture.embedTokensWeight());
        assertEquals("model.norm.weight", architecture.finalNormWeight());
        assertEquals(List.of("model.norm.weight", "model.final_layernorm.weight"),
                architecture.finalNormWeightCandidates());
        assertNull(architecture.finalNormBias());
        assertEquals("model.layers.5.self_attn.qkv_proj.weight", architecture.layerFusedQKVWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerQueryWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerKeyWeight(5));
        assertEquals(architecture.layerFusedQKVWeight(5), architecture.layerValueWeight(5));
        assertEquals("model.layers.5.self_attn.o_proj.weight", architecture.layerOutputWeight(5));
        assertEquals("model.layers.5.input_layernorm.weight", architecture.layerAttentionNormWeight(5));
        assertEquals("model.layers.5.mlp.gate_up_proj.weight", architecture.layerFfnGateWeight(5));
        assertEquals("model.layers.5.mlp.gate_up_proj.weight", architecture.layerFfnUpWeight(5));
        assertEquals("model.layers.5.mlp.down_proj.weight", architecture.layerFfnDownWeight(5));
        assertEquals("model.layers.5.post_attention_layernorm.weight", architecture.layerFfnNormWeight(5));
    }

    @Test
    void phiArchitectureClaimsPhiAndPhi3ConfigFamilies() {
        PhiFamily architecture = new PhiFamily();

        assertEquals(List.of("PhiForCausalLM", "Phi3ForCausalLM", "Phi3SmallForCausalLM"),
                architecture.supportedArchClassNames());
        assertEquals(List.of("phi", "phi3"), architecture.supportedModelTypes());
        assertTrue(architecture.usesRmsNorm());
    }

    private static Path fixture(String familyId) throws Exception {
        return Path.of(Objects.requireNonNull(
                PhiModelFamilyPluginTest.class.getResource("/model-family-fixtures/" + familyId)).toURI());
    }
}
