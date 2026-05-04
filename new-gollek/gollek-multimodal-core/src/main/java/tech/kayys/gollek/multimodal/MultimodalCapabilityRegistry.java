package tech.kayys.gollek.multimodal;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.model.MultimodalCapability;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for multimodal provider capabilities.
 */
@ApplicationScoped
public class MultimodalCapabilityRegistry {

    private final Map<String, MultimodalCapability> capabilities = new ConcurrentHashMap<>();
    private final Map<String, MultimodalInferenceProvider> providers = new ConcurrentHashMap<>();

    public void register(MultimodalInferenceProvider provider) {
        String id = provider.providerId();
        providers.put(id, provider);
        capabilities.put(id, provider.capability());
    }

    public Optional<MultimodalInferenceProvider> providerFor(String modelId) {
        return Optional.ofNullable(providers.get(modelId));
    }

    public List<MultimodalInferenceProvider> findCapable(Set<ModalityType> inputs, List<ModalityType> outputs) {
        // Simple filtering for now
        return providers.values().stream()
                .filter(p -> {
                    MultimodalCapability cap = p.capability();
                    return cap.getInputModalities().containsAll(inputs) &&
                            (outputs == null || cap.getOutputModalities().containsAll(outputs));
                })
                .toList();
    }

    public Map<String, MultimodalCapability> allCapabilities() {
        return Collections.unmodifiableMap(capabilities);
    }

    public Set<String> registeredIds() {
        return providers.keySet();
    }
}
