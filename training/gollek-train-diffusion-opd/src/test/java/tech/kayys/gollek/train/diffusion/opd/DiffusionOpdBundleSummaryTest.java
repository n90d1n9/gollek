package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiffusionOpdBundleSummaryTest {

    @Test
    void returnsTypedBundleSummaryForPublicApi() throws Exception {
        Path outputDir = Files.createTempDirectory("opd-bundle-summary");
        Files.writeString(outputDir.resolve("overview.json"), "{\"ok\":true}");
        Files.writeString(outputDir.resolve("tasks.csv"), "value,count\nocr,1\n");
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "sourceReportPath", "/tmp/diffusion-opd-report.json",
                "outputDirectory", outputDir.toString(),
                "createdAt", "2026-05-18T00:00:00Z",
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"),
                        Map.of("name", "tasks.csv", "section", "taskSummaries", "format", "csv"),
                        Map.of("name", "missing.txt", "section", "overview", "format", "text"))));

        DiffusionOpdBundleSummary summary = DiffusionOpdBundleInspector.summary(manifest);

        assertEquals("standard", summary.bundleType());
        assertEquals(3, summary.totalFiles());
        assertEquals(2, summary.existingFileCount());
        assertEquals(1, summary.missingFileCount());
        assertEquals("section", summary.largestSection().groupBy());
        assertEquals("overview", summary.largestSection().value());
        assertEquals("format", summary.largestFormat().groupBy());
        assertEquals("json", summary.largestFormat().value());
        assertEquals("all", summary.focus());
        assertFalse(summary.dominantOnly());
        assertTrue(summary.toMap().containsKey("sections"));
    }
}
