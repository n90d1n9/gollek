package tech.kayys.gollek.ml.hub;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class HubTest {

    @Test
    void testHubConfig() {
        var config = HubConfig.DEFAULT;
        assertNotNull(config);
    }

    @Test
    void testHubException() {
        var ex = new HubException("Test error");
        assertEquals("Test error", ex.getMessage());
    }

    @Test
    void testHubConfigBuilder() {
        var config = HubConfig.builder()
            .cacheDir("/tmp/cache")
            .build();
        assertNotNull(config);
    }
}
