package tech.kayys.gollek.provider.core;

import io.quarkus.vault.VaultKVSecretReactiveEngine;
import io.quarkus.vault.client.VaultException;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Manages secrets retrieval from HashiCorp Vault with caching and fallback support.
 */
@ApplicationScoped
public class VaultSecretManager {

    private static final Logger LOG = Logger.getLogger(VaultSecretManager.class);

    @Inject
    VaultKVSecretReactiveEngine kvReactiveEngine;

    @ConfigProperty(name = "gollek.vault.secret-prefix", defaultValue = "gollek/providers")
    String secretPrefix;

    @ConfigProperty(name = "gollek.vault.cache-enabled", defaultValue = "true")
    boolean cacheEnabled;

    @ConfigProperty(name = "gollek.vault.cache-ttl-minutes", defaultValue = "5")
    long cacheTtlMinutes;

    @ConfigProperty(name = "gollek.vault.fallback-enabled", defaultValue = "true")
    boolean fallbackEnabled;

    private final Map<String, CacheEntry> secretCache = new ConcurrentHashMap<>();

    public Map<String, String> getSecrets(String providerId) {
        String secretPath = buildSecretPath(providerId);
        
        if (cacheEnabled) {
            CacheEntry cached = secretCache.get(secretPath);
            if (cached != null && !cached.isExpired()) {
                return cached.secrets;
            }
        }

        try {
            Map<String, String> secrets = kvReactiveEngine.readSecret(secretPath)
                    .onFailure().recoverWithNull()
                    .await()
                    .atMost(Duration.ofSeconds(5));

            if (secrets == null || secrets.isEmpty()) {
                return Collections.emptyMap();
            }

            if (cacheEnabled) {
                secretCache.put(secretPath, new CacheEntry(secrets, cacheTtlMinutes));
            }

            return secrets;

        } catch (VaultException e) {
            LOG.errorf(e, "Vault error retrieving secrets for provider %s: %s", providerId, e.getMessage());
            return handleVaultError(providerId, e);
        } catch (Exception e) {
            LOG.errorf(e, "Unexpected error retrieving secrets for provider %s", providerId);
            return handleVaultError(providerId, e);
        }
    }

    public Optional<String> getSecret(String providerId, String secretKey) {
        Map<String, String> secrets = getSecrets(providerId);
        return Optional.ofNullable(secrets.get(secretKey));
    }

    public String getSecretOrDefault(String providerId, String secretKey, String defaultValue) {
        return getSecret(providerId, secretKey).orElse(defaultValue);
    }

    public void invalidateCache(String providerId) {
        String secretPath = buildSecretPath(providerId);
        secretCache.remove(secretPath);
    }

    public void invalidateAllCache() {
        secretCache.clear();
    }

    public boolean isVaultHealthy() {
        try {
            kvReactiveEngine.readSecret(secretPrefix + "/health-check")
                .await()
                .atMost(Duration.ofSeconds(2));
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private String buildSecretPath(String providerId) {
        return secretPrefix + "/" + providerId;
    }

    private Map<String, String> handleVaultError(String providerId, Exception e) {
        if (fallbackEnabled) {
            return getSecretsFromEnvironment(providerId);
        } else {
            return Collections.emptyMap();
        }
    }

    private Map<String, String> getSecretsFromEnvironment(String providerId) {
        Map<String, String> secrets = new HashMap<>();
        String envPrefix = "GOLLEK_PROVIDER_" + providerId.toUpperCase().replace('-', '_') + "_";
        
        System.getenv().entrySet().stream()
            .filter(entry -> entry.getKey().startsWith(envPrefix))
            .forEach(entry -> {
                String key = entry.getKey().substring(envPrefix.length()).toLowerCase();
                secrets.put(key, entry.getValue());
            });
        
        return secrets;
    }

    private static class CacheEntry {
        private final Map<String, String> secrets;
        private final long expiryTime;

        CacheEntry(Map<String, String> secrets, long ttlMinutes) {
            this.secrets = Collections.unmodifiableMap(new HashMap<>(secrets));
            this.expiryTime = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(ttlMinutes);
        }

        boolean isExpired() {
            return System.currentTimeMillis() > expiryTime;
        }
    }
}
