package tech.kayys.gollek.model.registry;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.spi.model.ModelFamilyCapability;
import tech.kayys.gollek.spi.model.ModelFamilyDescriptor;
import tech.kayys.gollek.spi.model.ModelFamilyPlugin;
import tech.kayys.gollek.spi.model.ModelFamilyPluginRegistry;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelArchitectureRegistryTest {

    @Test
    void prefersAdapterFromMatchedModelFamilyOverUnrelatedGlobalAdapter() throws Exception {
        ModelFamilyPlugin unrelated = plugin(
                "aaa-unrelated-adapter",
                "other_architecture_resolution",
                "OtherArchitectureResolutionForCausalLM",
                new TestArchitecture(
                        "unrelated-adapter",
                        "architecture_resolution",
                        "ArchitectureResolutionForCausalLM"));
        ModelFamilyPlugin matched = plugin(
                "zzz-matched-adapter",
                "architecture_resolution",
                "ArchitectureResolutionForCausalLM",
                new TestArchitecture(
                        "matched-adapter",
                        "architecture_resolution",
                        "ArchitectureResolutionForCausalLM"));
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "architecture_resolution",
                  "architectures": ["ArchitectureResolutionForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry.global().register(unrelated);
        ModelFamilyPluginRegistry.global().register(matched);
        try {
            ModelArchitecture resolved = new ModelArchitectureRegistry().resolve(config);

            assertEquals("matched-adapter", resolved.id());
        } finally {
            ModelFamilyPluginRegistry.global().unregister(unrelated.id());
            ModelFamilyPluginRegistry.global().unregister(matched.id());
        }
    }

    @Test
    void failureIncludesModelFamilyRemediationWhenNoFamilyMatches() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "architecture_missing_resolution",
                  "architectures": ["ArchitectureMissingResolutionForCausalLM"]
                }
                """, ModelConfig.class);

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> new ModelArchitectureRegistry().resolve(config));

        assertTrue(error.getMessage().contains("model_family_not_found"));
        assertTrue(error.getMessage().contains("architecture_missing_resolution"));
        assertTrue(error.getMessage().contains("Attach or install a model-family plugin"));
    }

    @Test
    void failureIncludesMatchedFamilySupportWhenNoAdapterMatches() throws Exception {
        ModelFamilyPlugin plugin = pluginWithoutAdapters(
                "architecture-no-adapter",
                "architecture_no_adapter",
                "ArchitectureNoAdapterForCausalLM");
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "architecture_no_adapter",
                  "architectures": ["ArchitectureNoAdapterForCausalLM"]
                }
                """, ModelConfig.class);

        ModelFamilyPluginRegistry.global().register(plugin);
        try {
            IllegalArgumentException error = assertThrows(
                    IllegalArgumentException.class,
                    () -> new ModelArchitectureRegistry().resolve(config));

            assertTrue(error.getMessage().contains("architecture-no-adapter"));
            assertTrue(error.getMessage().contains("declared_no_adapter"));
            assertTrue(error.getMessage().contains("No compatible architecture adapter"));
        } finally {
            ModelFamilyPluginRegistry.global().unregister(plugin.id());
        }
    }

    private static ModelFamilyPlugin plugin(
            String familyId,
            String modelType,
            String architectureClassName,
            ModelArchitecture adapter) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        familyId,
                        familyId,
                        List.of(modelType),
                        List.of(architectureClassName),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }

            @Override
            public List<ModelArchitecture> architectureAdapters() {
                return List.of(adapter);
            }
        };
    }

    private static ModelFamilyPlugin pluginWithoutAdapters(
            String familyId,
            String modelType,
            String architectureClassName) {
        return new ModelFamilyPlugin() {
            @Override
            public ModelFamilyDescriptor descriptor() {
                return new ModelFamilyDescriptor(
                        familyId,
                        familyId,
                        List.of(modelType),
                        List.of(architectureClassName),
                        List.of(ModelFamilyCapability.DIRECT_SAFETENSOR_INFERENCE),
                        Map.of("bundle_profile", "core"));
            }
        };
    }

    private record TestArchitecture(
            String id,
            String modelType,
            String architectureClassName) implements ModelArchitecture {

        @Override
        public List<String> supportedArchClassNames() {
            return List.of(architectureClassName);
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of(modelType);
        }

        @Override
        public String embedTokensWeight() {
            return "embed.weight";
        }

        @Override
        public String finalNormWeight() {
            return "norm.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "q.weight";
        }

        @Override
        public String layerKeyWeight(int i) {
            return "k.weight";
        }

        @Override
        public String layerValueWeight(int i) {
            return "v.weight";
        }

        @Override
        public String layerOutputWeight(int i) {
            return "o.weight";
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "attn_norm.weight";
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "gate.weight";
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "up.weight";
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "down.weight";
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "ffn_norm.weight";
        }
    }
}
