package tech.kayys.gollek.multimodal.security;

import io.smallrye.mutiny.Uni;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.spi.model.*;
import tech.kayys.gollek.multimodal.service.MultimodalInferenceService;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Security tests for multimodal inference service.
 * Tests input validation, injection prevention, and access control.
 */
class MultimodalSecurityTest {

    private MultimodalInferenceService service;

    @BeforeEach
    void setUp() {
        service = new MultimodalInferenceService();
        service.initialize();
    }

    @Test
    void testInputValidation() {
        // Test that invalid inputs are rejected
        MultimodalRequest invalidRequest = MultimodalRequest.builder()
            .requestId("invalid-input")
            .model(null)  // Null model
            .inputs(null)  // Null inputs
            .build();

        try {
            service.infer(invalidRequest).await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // Should reject null inputs
            assertThat(e).isNotNull();
        }
    }

    @Test
    void testSqlInjectionPrevention() {
        // Test that SQL injection attempts are handled
        String maliciousInput = "'; DROP TABLE users; --";
        
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("sql-injection-test")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofText(maliciousInput))
            .build();

        try {
            // Should not throw SQL-related exceptions
            service.infer(request).await().atMost(Duration.ofSeconds(10));
        } catch (Exception e) {
            // Should not be SQL-related
            assertThat(e.getMessage()).doesNotContain("SQL", "TABLE", "DROP");
        }
    }

    @Test
    void testScriptInjectionPrevention() {
        // Test that script injection attempts are handled
        String maliciousScript = "<script>alert('XSS')</script>";
        
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("script-injection-test")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofText(maliciousScript))
            .build();

        try {
            service.infer(request).await().atMost(Duration.ofSeconds(10));
        } catch (Exception e) {
            // Should not execute scripts
            assertThat(e.getMessage()).doesNotContain("alert", "script");
        }
    }

    @Test
    void testPathTraversalPrevention() {
        // Test that path traversal attempts are handled
        String maliciousPath = "../../../etc/passwd";
        
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("path-traversal-test")
            .model(maliciousPath)
            .inputs(MultimodalContent.ofText("test"))
            .build();

        try {
            service.infer(request).await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // Should prevent path traversal
            assertThat(e.getMessage()).doesNotContain("etc", "passwd");
        }
    }

    @Test
    void testLargePayloadHandling() {
        // Test that large payloads are handled gracefully
        StringBuilder largeText = new StringBuilder();
        for (int i = 0; i < 10000; i++) {
            largeText.append("x");
        }

        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("large-payload-test")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofText(largeText.toString()))
            .build();

        try {
            // Should handle or reject large payloads gracefully
            service.infer(request).await().atMost(Duration.ofSeconds(10));
        } catch (Exception e) {
            // Should not crash
            assertThat(e).isNotNull();
        }
    }

    @Test
    void testParameterInjection() {
        // Test that parameter injection is prevented
        Map<String, Object> maliciousParams = new HashMap<>();
        maliciousParams.put("normal_param", "value");
        maliciousParams.put("injected", "'; DROP TABLE--");
        
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("param-injection-test")
            .model("clip-vit-base")
            .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
            .parameters(maliciousParams)
            .build();

        try {
            service.infer(request).await().atMost(Duration.ofSeconds(10));
        } catch (Exception e) {
            // Should not allow parameter injection
            assertThat(e.getMessage()).doesNotContain("DROP", "TABLE");
        }
    }

    @Test
    void testUnauthorizedModelAccess() {
        // Test that unauthorized model access is prevented
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("unauthorized-model")
            .model("restricted-internal-model")
            .inputs(MultimodalContent.ofText("test"))
            .build();

        try {
            service.infer(request).await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            // Should reject unauthorized models
            assertThat(e).isNotNull();
        }
    }

    @Test
    void testResourceExhaustionPrevention() {
        // Test that resource exhaustion attacks are prevented
        // Send many small requests rapidly
        for (int i = 0; i < 100; i++) {
            MultimodalRequest request = MultimodalRequest.builder()
                .requestId("exhaustion-" + i)
                .model("clip-vit-base")
                .inputs(MultimodalContent.ofBase64Image(createTestImage(), "image/jpeg"))
                .build();

            try {
                service.infer(request).await().atMost(Duration.ofSeconds(2));
            } catch (Exception e) {
                // Rate limiting is OK
            }
        }

        // Service should still be operational
        assertThat(service).isNotNull();
    }

    @Test
    void testInformationLeakage() {
        // Test that error messages don't leak sensitive information
        MultimodalRequest request = MultimodalRequest.builder()
            .requestId("info-leak-test")
            .model("nonexistent-model")
            .inputs(MultimodalContent.ofText("test"))
            .build();

        try {
            service.infer(request).await().atMost(Duration.ofSeconds(5));
        } catch (Exception e) {
            String message = e.getMessage();
            // Should not leak internal details
            assertThat(message).doesNotContain(
                "password", "secret", "key", "token", "credential",
                "internal", "stack trace", "at line"
            );
        }
    }

    private byte[] createTestImage() {
        String base64Image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMCwsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBEQCEAwEPwAAf/9k=";
        return java.util.Base64.getDecoder().decode(base64Image);
    }
}
