package tech.kayys.gollek.server.api.v1;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.core.model.ModelFormat;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderInfo;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ModelCapabilityMapperTest {

    @Test
    @SuppressWarnings("unchecked")
    void buildsModelCapabilityMatrixFromModelAndProviderMetadata() {
        ModelInfo model = ModelInfo.builder()
                .modelId("demo-model")
                .format("GGUF")
                .architecture("llama")
                .contextLength(4096L)
                .embeddingSize(768)
                .outputTokenLimit(512)
                .metadata(Map.of("json_mode", true))
                .build();
        ProviderCapabilities capabilities = ProviderCapabilities.builder()
                .streaming(true)
                .functionCalling(true)
                .toolCalling(true)
                .structuredOutputs(true)
                .embeddings(true)
                .maxContextTokens(8192)
                .maxOutputTokens(1024)
                .features(Set.of("json_mode"))
                .supportedFormats(Set.of(ModelFormat.GGUF))
                .supportedDevices(Set.of(DeviceType.CPU))
                .supportedModels(Set.of("demo-model"))
                .build();
        ProviderInfo provider = ProviderInfo.builder()
                .id("local")
                .name("Local")
                .healthStatus(ProviderHealth.Status.HEALTHY)
                .capabilities(capabilities)
                .supportedModels(Set.of("demo-model"))
                .build();

        Map<String, Object> matrix = ModelCapabilityMapper.toCapabilityMatrix(
                "demo-model",
                Optional.of(model),
                List.of(provider),
                Optional.of("local"));

        assertEquals("demo-model", matrix.get("model_id"));
        assertEquals(true, matrix.get("known"));
        assertEquals(true, matrix.get("available"));
        assertEquals("local", matrix.get("preferred_provider"));

        Map<String, Object> limits = (Map<String, Object>) matrix.get("limits");
        assertEquals(4096L, limits.get("context_tokens"));
        assertEquals(512, limits.get("output_tokens"));
        assertEquals(768, limits.get("embedding_dimensions"));

        Map<String, Object> inference = (Map<String, Object>) matrix.get("inference");
        assertEquals(true, inference.get("streaming"));
        assertEquals(true, inference.get("json_mode"));
        assertEquals(true, inference.get("structured_outputs"));

        Map<String, Object> tooling = (Map<String, Object>) matrix.get("tooling");
        assertEquals(true, tooling.get("tool_definitions"));
        assertEquals(true, tooling.get("model_tool_calling"));
        assertEquals(false, tooling.get("tool_execution"));

        List<Map<String, Object>> candidates = (List<Map<String, Object>>) matrix.get("provider_candidates");
        assertEquals(1, candidates.size());
        assertEquals("local", candidates.get(0).get("id"));
    }
}
