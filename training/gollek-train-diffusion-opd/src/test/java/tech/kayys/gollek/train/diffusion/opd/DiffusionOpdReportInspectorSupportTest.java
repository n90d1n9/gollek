package tech.kayys.gollek.train.diffusion.opd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DiffusionOpdReportInspectorSupportTest {

    @Test
    void inspectsGeneratedFilesThroughTypedManifest() {
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"),
                        Map.of("name", "tasks.csv", "section", "taskSummaries", "format", "csv"))));

        DiffusionOpdBundleView view =
                DiffusionOpdBundleInspector.inspect(manifest, "files", "text");

        assertEquals("files", view.section());
        assertEquals("text", view.format());
        List<?> files = assertInstanceOf(List.class, view.value());
        assertEquals(2, files.size());
        Map<?, ?> first = assertInstanceOf(Map.class, files.getFirst());
        assertEquals("overview.json", first.get("name"));
    }

    @Test
    void promotesLoadfileStructuredContentToJson() throws Exception {
        Path outputDir = Files.createTempDirectory("opd-inspector-bundle");
        Files.writeString(outputDir.resolve("overview.json"), "{\"status\":\"ok\"}");
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "outputDirectory", outputDir.toString(),
                "generatedFiles", List.of(
                        Map.of("name", "overview.json", "section", "overview", "format", "json"))));

        DiffusionOpdBundleView view =
                DiffusionOpdBundleInspector.inspect(manifest, "loadfile:overview.json", "text");

        assertEquals("loadfile:overview.json", view.section());
        assertEquals("json", view.format());
        Map<?, ?> loaded = assertInstanceOf(Map.class, view.value());
        assertEquals("ok", loaded.get("status"));
    }

    @Test
    void summarizesBundleHealthFromTypedManifest() throws Exception {
        Path outputDir = Files.createTempDirectory("opd-inspector-health");
        DiffusionOpdBundleManifest manifest = DiffusionOpdBundleManifests.fromMap(Map.of(
                "bundleType", "standard",
                "outputDirectory", outputDir.toString(),
                "generatedFiles", List.of(
                        Map.of("name", "missing.json", "section", "overview", "format", "json"))));

        DiffusionOpdBundleView view =
                DiffusionOpdBundleInspector.inspect(manifest, "bundleHealth", "json");

        Map<?, ?> health = assertInstanceOf(Map.class, view.value());
        assertEquals("broken", health.get("status"));
        assertEquals(1, health.get("missingFileCount"));
        assertEquals(Boolean.TRUE, health.get("outputDirectoryExists"));
        assertTrue(health.containsKey("checks"));
    }
}
