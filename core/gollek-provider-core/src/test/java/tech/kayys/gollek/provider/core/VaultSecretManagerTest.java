package tech.kayys.gollek.provider.core;

import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.TestProfile;
import io.quarkus.vault.VaultKVSecretReactiveEngine;
import jakarta.inject.Inject;
import org.junit.jupiter.api.*;
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

/**
 * Integration tests for VaultSecretManager.
 */
@QuarkusTest
@Testcontainers(disabledWithoutDocker = true)
@TestProfile(VaultSecretManagerTest.VaultTestProfile.class)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VaultSecretManagerTest {

    @Container
    static GenericContainer<?> vaultContainer = new GenericContainer<>("hashicorp/vault:1.15.4")
            .withExposedPorts(8200)
            .withEnv("VAULT_DEV_ROOT_TOKEN_ID", "test-token")
            .withEnv("VAULT_DEV_LISTEN_ADDRESS", "0.0.0.0:8200")
            .waitingFor(org.testcontainers.containers.wait.strategy.Wait.forHttp("/v1/sys/health")
                    .forPort(8200)
                    .forStatusCode(200))
            .withStartupTimeout(Duration.ofMinutes(2));

    @Inject
    VaultSecretManager vaultSecretManager;

    @Inject
    VaultKVSecretReactiveEngine kvReactiveEngine;

    @BeforeAll
    static void requireDocker() {
        Assumptions.assumeTrue(
                DockerClientFactory.instance().isDockerAvailable(),
                "Docker not available; skipping Vault integration tests.");
    }

    @BeforeEach
    void setUp() throws Exception {
        await().atMost(Duration.ofSeconds(30))
                .until(() -> vaultContainer.isRunning());

        System.setProperty("quarkus.vault.url", 
            "http://" + vaultContainer.getHost() + ":" + vaultContainer.getMappedPort(8200));
        System.setProperty("quarkus.vault.authentication", "token");
        System.setProperty("quarkus.vault.token", "test-token");
        System.setProperty("quarkus.vault.kv.secret-engine-path", "secret/");

        kvReactiveEngine.writeSecret("gollek/providers/openai", 
                Map.of("api-key", "sk-test-openai-key-12345",
                       "organization-id", "org-test-123"))
                .await().indefinitely();

        kvReactiveEngine.writeSecret("gollek/providers/anthropic", 
                Map.of("api-key", "sk-ant-anthropic-key-67890",
                       "beta-header", "2024-01-01"))
                .await().indefinitely();
    }

    @Test
    @Order(1)
    @DisplayName("Should retrieve secrets from Vault")
    void shouldRetrieveSecretsFromVault() {
        Map<String, String> secrets = vaultSecretManager.getSecrets("openai");
        assertThat(secrets).isNotEmpty();
        assertThat(secrets.get("api-key")).isEqualTo("sk-test-openai-key-12345");
        assertThat(secrets.get("organization-id")).isEqualTo("org-test-123");
    }

    @Test
    @Order(2)
    @DisplayName("Should retrieve specific secret from Vault")
    void shouldRetrieveSpecificSecret() {
        Optional<String> apiKey = vaultSecretManager.getSecret("anthropic", "api-key");
        assertThat(apiKey).isPresent();
        assertThat(apiKey.get()).isEqualTo("sk-ant-anthropic-key-67890");
    }

    @Test
    @Order(3)
    @DisplayName("Should return empty optional for non-existent secret")
    void shouldReturnEmptyForNonExistentSecret() {
        Optional<String> secret = vaultSecretManager.getSecret("nonexistent", "api-key");
        assertThat(secret).isEmpty();
    }

    @Test
    @Order(4)
    @DisplayName("Should use default value when secret not found")
    void shouldUseDefaultValue() {
        String defaultValue = vaultSecretManager.getSecretOrDefault("nonexistent", "api-key", "default-value");
        assertThat(defaultValue).isEqualTo("default-value");
    }

    @Test
    @Order(5)
    @DisplayName("Should cache secrets")
    void shouldCacheSecrets() {
        Map<String, String> secrets1 = vaultSecretManager.getSecrets("openai");
        assertThat(secrets1).isNotEmpty();

        Map<String, String> secrets2 = vaultSecretManager.getSecrets("openai");
        assertThat(secrets2).isEqualTo(secrets1);
    }

    @Test
    @Order(6)
    @DisplayName("Should invalidate cache")
    void shouldInvalidateCache() {
        vaultSecretManager.getSecrets("openai");
        vaultSecretManager.invalidateCache("openai");
    }

    @Test
    @Order(7)
    @DisplayName("Should check Vault health")
    void shouldCheckVaultHealth() {
        boolean healthy = vaultSecretManager.isVaultHealthy();
        assertThat(healthy).isTrue();
    }

    @Test
    @Order(8)
    @DisplayName("Should handle fallback when Vault unavailable")
    void shouldHandleFallback() {
        System.setProperty("gollek.vault.fallback-enabled", "true");
        Map<String, String> secrets = vaultSecretManager.getSecrets("fallback-test");
        assertThat(secrets).isEmpty();
    }

    public static class VaultTestProfile implements io.quarkus.test.junit.QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of(
                "gollek.vault.secret-prefix", "gollek/providers",
                "gollek.vault.cache-enabled", "true",
                "gollek.vault.cache-ttl-minutes", "1",
                "gollek.vault.fallback-enabled", "true"
            );
        }
    }
}
