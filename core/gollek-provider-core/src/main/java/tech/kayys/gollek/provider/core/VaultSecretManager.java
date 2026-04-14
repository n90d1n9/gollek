package tech.kayys.gollek.provider.core;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;

/**
 * Manages secrets retrieval with fallback support.
 * Vault integration disabled for JBang compatibility.
 */
@ApplicationScoped
public class VaultSecretManager {

    private static final Logger LOG = Logger.getLogger(VaultSecretManager.class);

    @ConfigProperty(name = "gollek.vault.fallback-enabled", defaultValue = "true")
    boolean fallbackEnabled;

    public Map<String, String> getSecrets(String providerId) {
        LOG.debugf("Vault not available, using fallback for provider: %s", providerId);
        return handleVaultError(providerId, new RuntimeException("Vault extension not available"));
    }

    public Optional<String> getSecret(String providerId, String secretKey) {
        Map<String, String> secrets = getSecrets(providerId);
        return Optional.ofNullable(secrets.get(secretKey));
    }

    public String getSecretOrDefault(String providerId, String secretKey, String defaultValue) {
        return getSecret(providerId, secretKey).orElse(defaultValue);
    }

    public void invalidateCache(String providerId) {
        // No-op when vault is disabled
    }

    public void invalidateAllCache() {
        // No-op when vault is disabled
    }

    public boolean isVaultHealthy() {
        return false;
    }

    private Map<String, String> handleVaultError(String providerId, Exception e) {
        if (fallbackEnabled) {
            LOG.debugf("Using fallback secret management for provider: %s", providerId);
            // Return empty map - secrets should be provided via config or environment variables
            return Collections.emptyMap();
        }
        LOG.errorf(e, "Vault error and fallback disabled for provider: %s", providerId);
        return Collections.emptyMap();
    }
}
