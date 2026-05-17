package tech.kayys.gollek.provider.core;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderRegistry;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@ApplicationScoped
public class DefaultProviderRegistry implements ProviderRegistry {

    private final Instance<LLMProvider> discoveredProviders;
    private final Map<String, LLMProvider> providersById = new ConcurrentHashMap<>();
    private final Map<String, LLMProvider> providersByVersionKey = new ConcurrentHashMap<>();
    private final Map<String, GollekPlugin> owningPlugins = new ConcurrentHashMap<>();

    @Inject
    public DefaultProviderRegistry(Instance<LLMProvider> discoveredProviders) {
        this.discoveredProviders = discoveredProviders;
        discoverProviders();
    }

    @Override
    public synchronized void discoverProviders() {
        providersById.clear();
        providersByVersionKey.clear();
        discoveredProviders.stream()
                .filter(LLMProvider::isEnabled)
                .forEach(this::register);
    }

    @Override
    public synchronized void register(LLMProvider provider) {
        providersByVersionKey.put(versionKey(provider.id(), provider.version()), provider);
        providersById.putIfAbsent(provider.id(), provider);
    }

    @Override
    public synchronized void registerProviderFromPlugin(LLMProvider provider, GollekPlugin sourcePlugin) {
        register(provider);
        if (sourcePlugin != null) {
            owningPlugins.put(provider.id(), sourcePlugin);
            owningPlugins.put(versionKey(provider.id(), provider.version()), sourcePlugin);
        }
    }

    @Override
    public synchronized void unregister(String providerId) {
        providersById.remove(providerId);
        providersByVersionKey.keySet().removeIf(key -> key.startsWith(providerId + "::"));
        owningPlugins.keySet().removeIf(key -> key.equals(providerId) || key.startsWith(providerId + "::"));
    }

    @Override
    public synchronized void unregister(String providerId, String version) {
        providersByVersionKey.remove(versionKey(providerId, version));
        owningPlugins.remove(versionKey(providerId, version));
        refreshPrimaryProvider(providerId);
    }

    @Override
    public Optional<LLMProvider> getProvider(String providerId) {
        return Optional.ofNullable(providersById.get(providerId));
    }

    @Override
    public Optional<LLMProvider> getProvider(String providerId, String version) {
        return Optional.ofNullable(providersByVersionKey.get(versionKey(providerId, version)));
    }

    @Override
    public LLMProvider getProviderOrThrow(String providerId) {
        return getProvider(providerId)
                .orElseThrow(() -> new IllegalArgumentException("Provider not found: " + providerId));
    }

    @Override
    public Collection<LLMProvider> getAllProviders() {
        return List.copyOf(new LinkedHashSet<>(providersById.values()));
    }

    @Override
    public Collection<LLMProvider> getAllProviderVersions() {
        return List.copyOf(new LinkedHashSet<>(providersByVersionKey.values()));
    }

    @Override
    public boolean hasProvider(String providerId) {
        return providersById.containsKey(providerId);
    }

    @Override
    public List<LLMProvider> getProvidersForModel(String model) {
        return providersByVersionKey.values().stream()
                .filter(provider -> supportsModel(provider, model))
                .toList();
    }

    @Override
    public List<StreamingProvider> getStreamingProviders() {
        return providersByVersionKey.values().stream()
                .filter(StreamingProvider.class::isInstance)
                .map(StreamingProvider.class::cast)
                .toList();
    }

    @Override
    public Optional<GollekPlugin> getOwningPlugin(String providerId) {
        return Optional.ofNullable(owningPlugins.get(providerId));
    }

    @Override
    public synchronized void shutdown() {
        Set<LLMProvider> uniqueProviders = new LinkedHashSet<>(providersByVersionKey.values());
        uniqueProviders.forEach(provider -> {
            try {
                provider.shutdown();
            } catch (Exception ignored) {
                // Best-effort shutdown for CLI lifecycle; discovery should not fail on cleanup.
            }
        });
        providersById.clear();
        providersByVersionKey.clear();
        owningPlugins.clear();
    }

    private boolean supportsModel(LLMProvider provider, String model) {
        try {
            if (provider.id().equalsIgnoreCase(model) || provider.name().equalsIgnoreCase(model)) {
                return true;
            }
            if (provider.metadata().getDefaultModel() != null
                    && provider.metadata().getDefaultModel().equalsIgnoreCase(model)) {
                return true;
            }
            ProviderRequest probe = ProviderRequest.builder()
                    .model(model)
                    .timeout(Duration.ofSeconds(1))
                    .build();
            return provider.supports(model, probe);
        } catch (Exception ignored) {
            return false;
        }
    }

    private void refreshPrimaryProvider(String providerId) {
        List<LLMProvider> matches = new ArrayList<>();
        for (LLMProvider provider : providersByVersionKey.values()) {
            if (provider.id().equals(providerId)) {
                matches.add(provider);
            }
        }
        if (matches.isEmpty()) {
            providersById.remove(providerId);
        } else {
            providersById.put(providerId, matches.getFirst());
        }
    }

    private static String versionKey(String providerId, String version) {
        return providerId + "::" + version;
    }
}
