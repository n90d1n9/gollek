package tech.kayys.gollek.server.api.v1;

import tech.kayys.gollek.core.model.ModelFormat;
import tech.kayys.gollek.core.tensor.DeviceType;
import tech.kayys.gollek.spi.model.ModelInfo;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderInfo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

final class ModelCapabilityMapper {

    private ModelCapabilityMapper() {
    }

    static Map<String, Object> toCapabilityMatrix(String modelId, Optional<ModelInfo> model,
            List<ProviderInfo> providers, Optional<String> preferredProvider) {
        List<ProviderInfo> candidates = providerCandidates(modelId, providers, preferredProvider);
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("model_id", modelId);
        payload.put("known", model.isPresent());
        payload.put("available", model.isPresent() || !candidates.isEmpty());
        payload.put("preferred_provider", preferredProvider.orElse(null));
        model.ifPresent(info -> {
            payload.put("format", info.getFormat());
            payload.put("architecture", info.getArchitecture());
            payload.put("quantization", info.getQuantization());
        });

        payload.put("limits", limits(model, candidates));
        payload.put("api_contract", apiContract());
        payload.put("openai_compatibility", openAiCompatibility(model, candidates));
        payload.put("inference", inferenceCapabilities(model, candidates));
        payload.put("tooling", toolingCapabilities(model, candidates));
        payload.put("rag", ragCapabilities());
        payload.put("embeddings", embeddingCapabilities(model, candidates));
        payload.put("modalities", modalityCapabilities(model, candidates));
        payload.put("provider_candidates", providerSummaries(modelId, candidates));
        payload.put("metadata", model.map(ModelInfo::getMetadata).orElse(Map.of()));
        return payload;
    }

    private static Map<String, Object> limits(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        Map<String, Object> limits = new LinkedHashMap<>();
        limits.put("context_tokens", firstLong(
                model.map(ModelInfo::getContextLength).orElse(null),
                maxProviderContext(providers)));
        limits.put("input_tokens", model.map(ModelInfo::getInputTokenLimit).orElse(null));
        limits.put("output_tokens", firstInteger(
                model.map(ModelInfo::getOutputTokenLimit).orElse(null),
                maxProviderOutput(providers)));
        limits.put("embedding_dimensions", model.map(ModelInfo::getEmbeddingSize).orElse(null));
        return limits;
    }

    private static Map<String, Object> apiContract() {
        Map<String, Object> contract = new LinkedHashMap<>();
        contract.put("chat_completions", true);
        contract.put("chat_streaming", true);
        contract.put("responses", true);
        contract.put("responses_streaming", true);
        contract.put("embeddings_endpoint", true);
        contract.put("system_prompt", true);
        contract.put("tools_request_schema", true);
        contract.put("mcp_tool_definitions", true);
        contract.put("rag_context_injection", true);
        return contract;
    }

    private static Map<String, Object> openAiCompatibility(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        Map<String, Object> compat = new LinkedHashMap<>();
        boolean embeddings = supportsEmbeddings(model, providers);
        compat.put("models", true);
        compat.put("chat_completions", true);
        compat.put("chat_streaming", true);
        compat.put("responses", true);
        compat.put("responses_streaming", true);
        compat.put("embeddings", embeddings);
        return compat;
    }

    private static Map<String, Object> inferenceCapabilities(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        Map<String, Object> inference = new LinkedHashMap<>();
        inference.put("completion", model.isPresent() || !providers.isEmpty());
        inference.put("streaming", supportsStreaming(providers));
        inference.put("system_prompt", true);
        inference.put("json_mode", metadataFlag(model, "json_mode", "json", "response_format")
                || providersHaveFeature(providers, "json_mode", "json", "response_format"));
        inference.put("structured_outputs", metadataFlag(model, "structured_outputs", "structured_output")
                || anyCapability(providers, ProviderCapabilities::isStructuredOutputs));
        inference.put("provider_candidates_available", !providers.isEmpty());
        return inference;
    }

    private static Map<String, Object> toolingCapabilities(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        boolean modelToolCalls = metadataFlag(model, "tool_calling", "tools", "function_calling");
        boolean providerToolCalls = anyCapability(providers, caps -> caps.isToolCalling() || caps.isFunctionCalling());
        Map<String, Object> tooling = new LinkedHashMap<>();
        tooling.put("tool_definitions", true);
        tooling.put("tool_choice", true);
        tooling.put("tool_call_request_items", true);
        tooling.put("tool_result_input_items", true);
        tooling.put("model_tool_calling", modelToolCalls || providerToolCalls);
        tooling.put("mcp_tool_definitions", true);
        tooling.put("tool_execution", false);
        return tooling;
    }

    private static Map<String, Object> ragCapabilities() {
        Map<String, Object> rag = new LinkedHashMap<>();
        rag.put("context_injection", true);
        rag.put("sources_metadata", true);
        rag.put("embedding_model_hint", true);
        rag.put("retrieval_policy", false);
        rag.put("vector_store_ownership", false);
        return rag;
    }

    private static Map<String, Object> embeddingCapabilities(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        boolean supported = supportsEmbeddings(model, providers);
        Map<String, Object> embeddings = new LinkedHashMap<>();
        embeddings.put("generation", supported);
        embeddings.put("endpoint", "/v1/embeddings");
        embeddings.put("openai_compatible", supported);
        embeddings.put("dimensions", model.map(ModelInfo::getEmbeddingSize).orElse(null));
        embeddings.put("encoding_formats", List.of("float", "base64"));
        embeddings.put("input_aliases", List.of("input", "inputs"));
        embeddings.put("batch_inputs", true);
        embeddings.put("metadata_passthrough", true);
        embeddings.put("retrieval_policy", false);
        embeddings.put("vector_store_ownership", false);
        return embeddings;
    }

    private static Map<String, Object> modalityCapabilities(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        Map<String, Object> modalities = new LinkedHashMap<>();
        modalities.put("text", true);
        modalities.put("embeddings", supportsEmbeddings(model, providers));
        modalities.put("multimodal", metadataFlag(model, "multimodal", "vision", "image")
                || anyCapability(providers, ProviderCapabilities::isMultimodal));
        return modalities;
    }

    private static List<Map<String, Object>> providerSummaries(String modelId, List<ProviderInfo> providers) {
        List<Map<String, Object>> summaries = new ArrayList<>();
        for (ProviderInfo provider : providers) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", provider.id());
            item.put("name", provider.name());
            item.put("health", provider.healthStatus() != null
                    ? provider.healthStatus().name().toLowerCase(Locale.ROOT)
                    : "unknown");
            item.put("supports_model", supportsModel(provider, modelId));
            item.put("capabilities", providerCapabilitySummary(provider.capabilities()));
            summaries.add(item);
        }
        return summaries;
    }

    private static Map<String, Object> providerCapabilitySummary(ProviderCapabilities caps) {
        Map<String, Object> summary = new LinkedHashMap<>();
        if (caps == null) {
            return summary;
        }
        summary.put("streaming", caps.isStreaming());
        summary.put("function_calling", caps.isFunctionCalling());
        summary.put("tool_calling", caps.isToolCalling());
        summary.put("structured_outputs", caps.isStructuredOutputs());
        summary.put("embeddings", caps.isEmbeddings());
        summary.put("multimodal", caps.isMultimodal());
        summary.put("max_context_tokens", caps.getMaxContextTokens());
        summary.put("max_output_tokens", caps.getMaxOutputTokens());
        summary.put("features", caps.getFeatures());
        summary.put("formats", caps.getSupportedFormats().stream().map(ModelFormat::name).sorted().toList());
        summary.put("devices", caps.getSupportedDevices().stream().map(DeviceType::name).sorted().toList());
        return summary;
    }

    private static List<ProviderInfo> providerCandidates(String modelId, List<ProviderInfo> providers,
            Optional<String> preferredProvider) {
        if (providers == null || providers.isEmpty()) {
            return List.of();
        }
        return providers.stream()
                .filter(provider -> preferredProvider.map(id -> id.equals(provider.id())).orElse(false)
                        || supportsModel(provider, modelId))
                .sorted(Comparator.comparing((ProviderInfo p) -> preferredProvider.map(id -> !id.equals(p.id())).orElse(true))
                        .thenComparing(p -> p.id() == null ? "" : p.id()))
                .toList();
    }

    private static boolean supportsModel(ProviderInfo provider, String modelId) {
        if (provider == null) {
            return false;
        }
        Set<String> providerModels = provider.supportedModels();
        if (providerModels != null && (providerModels.isEmpty() || providerModels.contains(modelId))) {
            return true;
        }
        ProviderCapabilities caps = provider.capabilities();
        return caps != null && caps.supportsModel(modelId);
    }

    private static boolean supportsStreaming(List<ProviderInfo> providers) {
        return anyCapability(providers, ProviderCapabilities::isStreaming);
    }

    private static boolean supportsEmbeddings(Optional<ModelInfo> model, List<ProviderInfo> providers) {
        return model.map(ModelInfo::getEmbeddingSize).orElse(null) != null
                || metadataFlag(model, "embeddings", "embedding", "embedding_generation")
                || model.map(ModelInfo::getModelId).map(ModelCapabilityMapper::looksLikeEmbeddingModel).orElse(false)
                || anyCapability(providers, ProviderCapabilities::isEmbeddings);
    }

    private static boolean looksLikeEmbeddingModel(String modelId) {
        String normalized = modelId.toLowerCase(Locale.ROOT);
        return normalized.contains("embed") || normalized.contains("embedding");
    }

    private static boolean providersHaveFeature(List<ProviderInfo> providers, String... names) {
        for (ProviderInfo provider : providers) {
            ProviderCapabilities caps = provider.capabilities();
            if (caps == null) {
                continue;
            }
            for (String name : names) {
                if (caps.hasFeature(name) || caps.getFeatures().stream()
                        .anyMatch(feature -> normalizeFeature(feature).equals(normalizeFeature(name)))) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean anyCapability(List<ProviderInfo> providers,
            java.util.function.Predicate<ProviderCapabilities> predicate) {
        for (ProviderInfo provider : providers) {
            ProviderCapabilities caps = provider.capabilities();
            if (caps != null && predicate.test(caps)) {
                return true;
            }
        }
        return false;
    }

    private static boolean metadataFlag(Optional<ModelInfo> model, String... keys) {
        return model.map(ModelInfo::getMetadata).map(metadata -> metadataFlag(metadata, keys)).orElse(false);
    }

    private static boolean metadataFlag(Map<String, Object> metadata, String... keys) {
        if (metadata == null || metadata.isEmpty()) {
            return false;
        }
        for (String key : keys) {
            Object direct = metadata.get(key);
            if (truthy(direct)) {
                return true;
            }
        }
        Object capabilities = metadata.get("capabilities");
        if (containsAny(capabilities, keys)) {
            return true;
        }
        Object features = metadata.get("features");
        return containsAny(features, keys);
    }

    private static boolean containsAny(Object value, String... keys) {
        if (value instanceof Map<?, ?> map) {
            for (String key : keys) {
                if (truthy(map.get(key))) {
                    return true;
                }
            }
            return false;
        }
        if (value instanceof Collection<?> collection) {
            for (Object item : collection) {
                String normalized = normalizeFeature(item);
                for (String key : keys) {
                    if (normalizeFeature(key).equals(normalized)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static boolean truthy(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        if (value instanceof String text) {
            return "true".equalsIgnoreCase(text) || "yes".equalsIgnoreCase(text) || "supported".equalsIgnoreCase(text);
        }
        return false;
    }

    private static String normalizeFeature(Object value) {
        return value == null ? "" : value.toString().trim().toLowerCase(Locale.ROOT).replace('-', '_');
    }

    private static Long firstLong(Long first, Integer second) {
        return first != null ? first : second != null && second > 0 ? second.longValue() : null;
    }

    private static Integer firstInteger(Integer first, Integer second) {
        return first != null ? first : second != null && second > 0 ? second : null;
    }

    private static Integer maxProviderContext(List<ProviderInfo> providers) {
        return providers.stream()
                .map(ProviderInfo::capabilities)
                .filter(caps -> caps != null && caps.getMaxContextTokens() > 0)
                .map(ProviderCapabilities::getMaxContextTokens)
                .max(Integer::compareTo)
                .orElse(null);
    }

    private static Integer maxProviderOutput(List<ProviderInfo> providers) {
        return providers.stream()
                .map(ProviderInfo::capabilities)
                .filter(caps -> caps != null && caps.getMaxOutputTokens() > 0)
                .map(ProviderCapabilities::getMaxOutputTokens)
                .max(Integer::compareTo)
                .orElse(null);
    }
}
