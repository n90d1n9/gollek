package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiffusionOpdBundleInspectorTest {

    @Test
    void returnsTypedBundleHealthForPublicApi() throws Exception {
        Path outputDir = Files.createTempDirectory("opd-bundle-inspector-health");
        Files.writeString(outputDir.resolve("overview.json"), "{\"ok\":true}");
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "outputDirectory", outputDir.toString(),
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"),
                        Map.of("name", "missing.csv", "section", "taskSummaries", "format", "csv"))));

        DiffusionOpdBundleHealth health = DiffusionOpdBundleInspector.health(manifest);

        assertEquals("standard", health.bundleType());
        assertEquals("degraded", health.status());
        assertEquals(2, health.totalFiles());
        assertEquals(1, health.missingFileCount());
        assertEquals("warning", health.alertLevel());
        assertFalse(health.healthy());
        assertEquals("fail", health.healthBadge().checkStatus());
        assertEquals(1, health.checkSummary().failed());
        assertEquals("taskSummaries", health.missingSections().getFirst().value());
        assertEquals("csv", health.missingFormats().getFirst().value());
        assertTrue(health.toMap().containsKey("checkSummary"));
    }
}
