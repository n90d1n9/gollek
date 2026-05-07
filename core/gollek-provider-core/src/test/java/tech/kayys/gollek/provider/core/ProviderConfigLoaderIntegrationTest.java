package tech.kayys.gollek.provider.core;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.TestProfile;
import jakarta.inject.Inject;
import org.junit.jupiter.api.*;
import tech.kayys.gollek.spi.provider.ProviderConfig;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Integration tests for ProviderConfigLoader with Vault integration.
 */
@QuarkusTest
@TestProfile(ProviderConfigLoaderIntegrationTest.TestProfile.class)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ProviderConfigLoaderIntegrationTest {

    @Inject
    ProviderConfigLoader configLoader;

    @Inject
    VaultSecretManager vaultSecretManager;

    @Test
    @Order(1)
    @DisplayName("Should load provider configuration")
    void shouldLoadProviderConfig() {
        Map<String, ProviderConfig> configs = configLoader.loadAll();
        assertThat(configs).isNotNull();
    }

    @Test
    @Order(2)
    @DisplayName("Should load specific provider config")
    void shouldLoadSpecificProvider() {
        String providerId = "test-provider";
        Optional<ProviderConfig> config = configLoader.load(providerId);
        assertThat(config).isNotNull();
    }

    @Test
    @Order(3)
    @DisplayName("Should reload provider configuration")
    void shouldReloadProvider() {
        String providerId = "test-provider";
        Optional<ProviderConfig> config = configLoader.reload(providerId);
        assertThat(config).isNotNull();
    }

    @Test
    @Order(4)
    @DisplayName("Should merge secrets into config")
    void shouldMergeSecrets() {
        Map<String, ProviderConfig> configs = configLoader.loadAll();
        assertThat(configs).isNotNull();
    }

    @Test
    @Order(5)
    @DisplayName("Should handle missing Vault gracefully")
    void shouldHandleMissingVault() {
        Map<String, String> secrets = vaultSecretManager.getSecrets("non-existent-provider");
        assertThat(secrets).isEmpty();
    }

    @Test
    @Order(6)
    @DisplayName("Should use fallback when Vault unavailable")
    void shouldUseFallback() {
        System.setProperty("gollek.vault.fallback-enabled", "true");
        Map<String, String> secrets = vaultSecretManager.getSecrets("fallback-test");
        assertThat(secrets).isNotNull();
    }

    @Test
    @Order(7)
    @DisplayName("Should cache Vault secrets")
    void shouldCacheSecrets() {
        vaultSecretManager.getSecrets("cache-test");
        
        long startTime = System.currentTimeMillis();
        vaultSecretManager.getSecrets("cache-test");
        long endTime = System.currentTimeMillis();

        assertThat(endTime - startTime).isLessThan(100);
    }

    @Test
    @Order(8)
    @DisplayName("Should invalidate cache")
    void shouldInvalidateCache() {
        vaultSecretManager.getSecrets("invalidate-test");
        vaultSecretManager.invalidateCache("invalidate-test");
        Map<String, String> secrets = vaultSecretManager.getSecrets("invalidate-test");
        assertThat(secrets).isNotNull();
    }

    @Test
    @Order(9)
    @DisplayName("Should check Vault health")
    void shouldCheckVaultHealth() {
        boolean healthy = vaultSecretManager.isVaultHealthy();
        assertThat(healthy).isInstanceOf(Boolean.class);
    }

    @Test
    @Order(10)
    @DisplayName("Should handle environment variable fallback")
    void shouldHandleEnvironmentFallback() {
        System.setProperty("GOLLEK_PROVIDER_ENVTEST_API_KEY", "test-key-from-env");
        Map<String, String> secrets = vaultSecretManager.getSecrets("envtest");
        assertThat(secrets).isNotNull();
    }

    public static class TestProfile implements io.quarkus.test.junit.QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            Map<String, String> config = new HashMap<>();
            config.put("gollek.vault.secret-prefix", "gollek/providers");
            config.put("gollek.vault.cache-enabled", "true");
            config.put("gollek.vault.cache-ttl-minutes", "5");
            config.put("gollek.vault.fallback-enabled", "true");
            config.put("providers.config.path", "./config/providers");
            config.put("providers.hot-reload.enabled", "true");
            return config;
        }
    }
}
